"""Numba blackjack table simulator with tabular Q-learning. Every seat at the table is a
counting agent playing and learning into one shared table (self-play).

Q[count_bucket, dealer_up - 1, hand_index, two_cards, action], rewards in initial bets.
Qi[count_bucket]: expected result of the insurance side bet when the dealer shows an ace.
hand_index: 0-17 hard 4-21, 18-27 soft 12-21, 28-37 splittable pairs A,A 2,2 .. 10,10.
"""
import numpy as np
from numba import njit, prange

STAND, HIT, DOUBLE, SPLIT, SURRENDER = 0, 1, 2, 3, 4
N_ACTIONS = 5
ACTION_NAMES = ("stand", "hit", "double", "split", "surrender")
N_HAND = 38
N_UP = 10
MAX_STEPS = 24
MAX_HANDS = 4
MAX_SEATS = 7

HAND_LABELS = (
    [f"H{t}" for t in range(4, 22)]
    + [f"S{t}" for t in range(12, 22)]
    + ["A,A"] + [f"{r},{r}" for r in range(2, 11)]
)


@njit(cache=True)
def hand_index(hard_sum, has_ace, pair_rank):
    if pair_rank > 0:
        return 28 + pair_rank - 1
    if has_ace and hard_sum + 10 <= 21:
        return 18 + hard_sum - 2
    return hard_sum - 4


@njit(cache=True)
def _total(hard_sum, has_ace):
    if has_ace and hard_sum + 10 <= 21:
        return hard_sum + 10
    return hard_sum


# st: [shoe position, running count in tag units]
# cp: [tag scale, true-count mode, initial running count, first bucket edge, bucket width, n buckets]
@njit(cache=True)
def _new_shoe(n_decks):
    shoe = np.empty(52 * n_decks, np.int64)
    k = 0
    for _ in range(4 * n_decks):
        for r in range(1, 14):
            shoe[k] = min(r, 10)
            k += 1
    return shoe


@njit(cache=True)
def _shuffle(shoe, st):
    np.random.shuffle(shoe)
    st[0] = 0
    st[1] = 0


@njit(cache=True)
def _draw(shoe, st, tags):
    if shoe.shape[0] == 0:
        return min(np.random.randint(1, 14), 10)
    if st[0] >= shoe.shape[0]:
        _shuffle(shoe, st)
    c = shoe[st[0]]
    st[0] += 1
    st[1] += tags[c]
    return c


@njit(cache=True)
def _count_bucket(shoe, st, hidden, cp):
    nb = int(cp[5])
    if nb == 1:
        return 0
    value = 0.0
    if shoe.shape[0] > 0:
        rc = st[1] / cp[0]
        if cp[1] > 0:
            value = rc * 52.0 / (shoe.shape[0] - st[0] + hidden)
        else:
            value = rc + cp[2]
    b = int(np.floor((value - cp[3]) / cp[4]))
    return min(max(b, 0), nb - 1)


@njit(cache=True)
def _vmax(Q, s, mask):
    best = -np.inf
    for a in range(N_ACTIONS):
        if (mask >> a) & 1:
            q = Q[s[0], s[1], s[2], s[3], a]
            if q > best:
                best = q
    return best


@njit(cache=True)
def _choose(Q, b, u, hidx, two, mask, eps):
    if eps > 0.0 and np.random.random() < eps:
        n = 0
        for a in range(N_ACTIONS):
            n += (mask >> a) & 1
        k = np.random.randint(0, n)
        for a in range(N_ACTIONS):
            if (mask >> a) & 1:
                if k == 0:
                    return a
                k -= 1
    best_a = STAND
    best = -np.inf
    for a in range(N_ACTIONS):
        if (mask >> a) & 1:
            q = Q[b, u, hidx, two, a]
            if q > best:
                best = q
                best_a = a
    return best_a


@njit(cache=True)
def _play_seat(shoe, st, Q, eps, tags, cp, u, das, dbl_min, max_hands, resplit_aces, hit_split_aces,
               surrender, c1, c2, h_sum, h_ace, h_n, h_c1, h_c2, h_bet, h_stat, tr_s, tr_a, tr_m, tr_c,
               tr_len):
    """Play one seat's hands (arrays are that seat's rows). Returns the number of hands."""
    for i in range(MAX_HANDS):
        tr_len[i] = 0
        h_bet[i] = 1.0
        h_stat[i] = 0  # 0 live, 1 bust, 2 surrendered
    h_sum[0] = c1 + c2
    h_ace[0] = c1 == 1 or c2 == 1
    h_n[0] = 2
    h_c1[0] = c1
    h_c2[0] = c2
    nh = 1
    h = 0
    while h < nh:
        while True:
            if h_n[h] == 1:  # split hand: second card
                c = _draw(shoe, st, tags)
                h_sum[h] += c
                h_ace[h] = h_ace[h] or c == 1
                h_n[h] = 2
                h_c2[h] = c
            t = _total(h_sum[h], h_ace[h])
            if t > 21:
                h_stat[h] = 1
                break
            if t == 21:
                break
            two = 1 if h_n[h] == 2 else 0
            split_hand = nh > 1
            ace_split = split_hand and h_c1[h] == 1
            can_split = (two == 1 and h_c1[h] == h_c2[h] and nh < max_hands
                         and (resplit_aces or not ace_split))
            if ace_split and not hit_split_aces:
                if not can_split:
                    break
                mask = (1 << STAND) | (1 << SPLIT)
            else:
                mask = (1 << STAND) | (1 << HIT)
                soft = h_ace[h] and h_sum[h] + 10 <= 21
                if two == 1 and (das or not split_hand) and (dbl_min == 0 or (not soft and dbl_min <= t <= 11)):
                    mask |= 1 << DOUBLE
                if can_split:
                    mask |= 1 << SPLIT
                if surrender and two == 1 and not split_hand:
                    mask |= 1 << SURRENDER
            b = _count_bucket(shoe, st, 1, cp)
            hidx = hand_index(h_sum[h], h_ace[h], h_c1[h] if can_split else 0)
            a = _choose(Q, b, u, hidx, two, mask, eps)
            k = tr_len[h]
            tr_s[h, k, 0] = b
            tr_s[h, k, 1] = u
            tr_s[h, k, 2] = hidx
            tr_s[h, k, 3] = two
            tr_a[h, k] = a
            tr_m[h, k] = mask
            tr_len[h] = k + 1
            if a == STAND:
                break
            elif a == HIT or a == DOUBLE:
                c = _draw(shoe, st, tags)
                h_sum[h] += c
                h_ace[h] = h_ace[h] or c == 1
                h_n[h] += 1
                if a == DOUBLE:
                    h_bet[h] = 2.0
                    if _total(h_sum[h], h_ace[h]) > 21:
                        h_stat[h] = 1
                    break
            elif a == SURRENDER:
                h_stat[h] = 2
                break
            else:  # SPLIT: the second card starts a new hand; both draw a new second card
                j = nh
                nh += 1
                r = h_c1[h]
                h_sum[j] = r
                h_ace[j] = r == 1
                h_n[j] = 1
                h_c1[j] = r
                h_sum[h] = r
                h_ace[h] = r == 1
                h_n[h] = 1
                tr_c[h, k] = j
        h += 1
    return nh


@njit(cache=True)
def _learn(Q, N, R, nh, tr_s, tr_a, tr_m, tr_c, tr_len):
    for h in range(nh):
        L = tr_len[h]
        for k in range(L - 1, -1, -1):
            a = tr_a[h, k]
            target = _vmax(Q, tr_s[h, k + 1], tr_m[h, k + 1]) if k + 1 < L else R[h]
            if a == SPLIT:  # plus the value of the hand split off
                j = tr_c[h, k]
                target += _vmax(Q, tr_s[j, 0], tr_m[j, 0]) if tr_len[j] > 0 else R[j]
            i0, i1, i2, i3 = tr_s[h, k, 0], tr_s[h, k, 1], tr_s[h, k, 2], tr_s[h, k, 3]
            n = N[i0, i1, i2, i3, a] + 1
            N[i0, i1, i2, i3, a] = n
            Q[i0, i1, i2, i3, a] += (target - Q[i0, i1, i2, i3, a]) / n


@njit(cache=True)
def _round(shoe, st, Q, N, Qi, Ni, learn, eps, tags, cp, cut, h17, das, dbl_min, max_hands,
           resplit_aces, hit_split_aces, peek, surrender, insurance, bj_payout, n_seats, rewards,
           c1, c2, nh, h_sum, h_ace, h_n, h_c1, h_c2, h_bet, h_stat, tr_s, tr_a, tr_m, tr_c, tr_len):
    """One round where every seat plays (and learns into) the shared table. Fills
    rewards[seat] and returns the count bucket before the deal."""
    if shoe.shape[0] > 0 and st[0] >= cut:
        _shuffle(shoe, st)
    pre_b = _count_bucket(shoe, st, 0, cp)

    for s in range(n_seats):
        c1[s] = _draw(shoe, st, tags)
    up = _draw(shoe, st, tags)
    for s in range(n_seats):
        c2[s] = _draw(shoe, st, tags)
    hole = _draw(shoe, st, tags)
    st[1] -= tags[hole]
    dealer_bj = (up == 1 and hole == 10) or (up == 10 and hole == 1)

    # insurance: half-bet side bet paying 2:1 on a dealer blackjack, decided on the count
    ins = 0.0
    if insurance and up == 1:
        bi = _count_bucket(shoe, st, 1, cp)
        outcome = 1.0 if dealer_bj else -0.5
        if learn:
            Ni[bi] += 1
            Qi[bi] += (outcome - Qi[bi]) / Ni[bi]
        if Qi[bi] > 0.0:
            ins = outcome
    for s in range(n_seats):
        rewards[s] = ins
        nh[s] = 0

    if peek and dealer_bj:
        st[1] += tags[hole]
        for s in range(n_seats):
            bj = (c1[s] == 1 and c2[s] == 10) or (c1[s] == 10 and c2[s] == 1)
            rewards[s] += 0.0 if bj else -1.0
        return pre_b

    any_live = False
    for s in range(n_seats):
        if (c1[s] == 1 and c2[s] == 10) or (c1[s] == 10 and c2[s] == 1):
            rewards[s] += 0.0 if dealer_bj else bj_payout
            continue
        nh[s] = _play_seat(shoe, st, Q, eps, tags, cp, up - 1, das, dbl_min, max_hands, resplit_aces,
                           hit_split_aces, surrender, c1[s], c2[s], h_sum[s], h_ace[s], h_n[s], h_c1[s],
                           h_c2[s], h_bet[s], h_stat[s], tr_s[s], tr_a[s], tr_m[s], tr_c[s], tr_len[s])
        for h in range(nh[s]):
            if h_stat[s, h] == 0:
                any_live = True

    st[1] += tags[hole]
    d_sum = up + hole
    d_ace = up == 1 or hole == 1
    if any_live and not dealer_bj:
        while True:
            t = _total(d_sum, d_ace)
            soft = d_ace and d_sum + 10 <= 21
            if t > 17 or (t == 17 and not (h17 and soft)):
                break
            c = _draw(shoe, st, tags)
            d_sum += c
            d_ace = d_ace or c == 1
    d_tot = _total(d_sum, d_ace)

    R = np.zeros(MAX_HANDS, np.float64)
    for s in range(n_seats):
        for h in range(nh[s]):
            if h_stat[s, h] == 2:
                R[h] = -0.5
            elif h_stat[s, h] == 1 or dealer_bj:
                R[h] = -h_bet[s, h]
            else:
                t = _total(h_sum[s, h], h_ace[s, h])
                if d_tot > 21 or t > d_tot:
                    R[h] = h_bet[s, h]
                elif t < d_tot:
                    R[h] = -h_bet[s, h]
                else:
                    R[h] = 0.0
            rewards[s] += R[h]
        if learn:
            _learn(Q, N, R, nh[s], tr_s[s], tr_a[s], tr_m[s], tr_c[s], tr_len[s])
    return pre_b


@njit(cache=True)
def _run_worker(Q, N, Qi, Ni, stats, n_rounds, seed, learn, eps, tags, cp, n_decks, penetration,
                h17, das, dbl_min, max_hands, resplit_aces, hit_split_aces, peek, surrender, insurance,
                bj_payout, n_seats):
    np.random.seed(seed)
    shoe = _new_shoe(n_decks)
    st = np.zeros(2, np.int64)
    if n_decks > 0:
        _shuffle(shoe, st)
    cut = int(penetration * shoe.shape[0])
    S, H = MAX_SEATS, MAX_HANDS
    rewards = np.zeros(S)
    c1 = np.zeros(S, np.int64)
    c2 = np.zeros(S, np.int64)
    nh = np.zeros(S, np.int64)
    h_sum = np.zeros((S, H), np.int64)
    h_ace = np.zeros((S, H), np.bool_)
    h_n = np.zeros((S, H), np.int64)
    h_c1 = np.zeros((S, H), np.int64)
    h_c2 = np.zeros((S, H), np.int64)
    h_bet = np.ones((S, H))
    h_stat = np.zeros((S, H), np.int64)
    tr_s = np.zeros((S, H, MAX_STEPS, 4), np.int64)
    tr_a = np.zeros((S, H, MAX_STEPS), np.int64)
    tr_m = np.zeros((S, H, MAX_STEPS), np.int64)
    tr_c = np.zeros((S, H, MAX_STEPS), np.int64)
    tr_len = np.zeros((S, H), np.int64)
    for _ in range(n_rounds):
        b = _round(shoe, st, Q, N, Qi, Ni, learn, eps, tags, cp, cut, h17, das, dbl_min, max_hands,
                   resplit_aces, hit_split_aces, peek, surrender, insurance, bj_payout, n_seats, rewards,
                   c1, c2, nh, h_sum, h_ace, h_n, h_c1, h_c2, h_bet, h_stat, tr_s, tr_a, tr_m, tr_c, tr_len)
        for s in range(n_seats):
            stats[s, b, 0] += 1.0
            stats[s, b, 1] += rewards[s]
            stats[s, b, 2] += rewards[s] * rewards[s]


@njit(parallel=True, cache=True)
def run_parallel(Qs, Ns, Qis, Nis, stats, n_rounds, seed, learn, eps, tags, cp, n_decks, penetration,
                 h17, das, dbl_min, max_hands, resplit_aces, hit_split_aces, peek, surrender, insurance,
                 bj_payout, n_seats):
    """One independent table per worker, each with its own shoe, RNG stream and Q copy.
    stats[w, seat, count bucket] = (rounds, sum of results, sum of squared results)."""
    for w in prange(Qs.shape[0]):
        _run_worker(Qs[w], Ns[w], Qis[w], Nis[w], stats[w], n_rounds, seed + 7919 * w, learn, eps,
                    tags, cp, n_decks, penetration, h17, das, dbl_min, max_hands, resplit_aces,
                    hit_split_aces, peek, surrender, insurance, bj_payout, n_seats)
