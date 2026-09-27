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
ST_SIZE = 12
N_FEAT = 11  # constant + excess of each rank left per deck (A, 2..9, T)

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


# st: [shoe position, running count in tag units, cards seen of rank 1 (ace) .. 10]
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
    for i in range(st.shape[0]):
        st[i] = 0


@njit(cache=True)
def _draw(shoe, st, tags):
    if shoe.shape[0] == 0:
        return min(np.random.randint(1, 14), 10)
    if st[0] >= shoe.shape[0]:
        _shuffle(shoe, st)
    c = shoe[st[0]]
    st[0] += 1
    st[1] += tags[c]
    st[1 + c] += 1
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
def _features(shoe, st, hidden, x):
    """x[0] = 1; x[r] = excess of rank r among the unseen cards, per deck (0 for a neutral shoe)."""
    unseen = shoe.shape[0] - st[0] + hidden
    n_decks = shoe.shape[0] // 52
    x[0] = 1.0
    for r in range(1, 11):
        per_deck = 16.0 if r == 10 else 4.0
        x[r] = 52.0 * (n_decks * per_deck - st[1 + r]) / unseen - per_deck


@njit(cache=True)
def _act(v, kind):
    if kind == 1:
        return v / (1.0 + np.exp(-v))  # SiLU
    return v if v > 0.0 else 0.0     # ReLU


@njit(cache=True)
def _deep_advantages(mlp, mdims, u, hidx, two, x, out):
    """Advantage outputs of the network part (two hidden layers) for one decision.
    Inputs: one-hot up card, one-hot hand, two-card flag, composition features x[1:]."""
    h1, h2, kind = mdims[0], mdims[1], mdims[2]
    n_in = N_UP + N_HAND + N_FEAT
    o = 0
    W1 = mlp[o:o + n_in * h1].reshape((n_in, h1))
    o += n_in * h1
    b1 = mlp[o:o + h1]
    o += h1
    W2 = mlp[o:o + h1 * h2].reshape((h1, h2))
    o += h1 * h2
    b2 = mlp[o:o + h2]
    o += h2
    W3 = mlp[o:o + h2 * (1 + N_ACTIONS)].reshape((h2, 1 + N_ACTIONS))
    o += h2 * (1 + N_ACTIONS)
    b3 = mlp[o:o + 1 + N_ACTIONS]
    y1 = np.empty(h1)
    for j in range(h1):
        v = b1[j] + W1[u, j] + W1[N_UP + hidx, j] + two * W1[N_UP + N_HAND, j]
        for f in range(1, N_FEAT):
            v += W1[N_UP + N_HAND + f, j] * x[f]
        y1[j] = _act(v, kind)
    y2 = np.empty(h2)
    for j in range(h2):
        v = b2[j]
        for i in range(h1):
            v += y1[i] * W2[i, j]
        y2[j] = _act(v, kind)
    for a in range(N_ACTIONS):
        v = b3[1 + a]
        for j in range(h2):
            v += y2[j] * W3[j, 1 + a]
        out[a] = v


@njit(cache=True)
def _choose_model(W, mlp, mdims, lin, u, hidx, two, mask, eps, x):
    """Best legal action of the linear model (lin 1), plus the network part (lin 2)."""
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
    deep = np.zeros(N_ACTIONS)
    if lin == 2:
        _deep_advantages(mlp, mdims, u, hidx, two, x, deep)
    best_a = STAND
    best = -np.inf
    for a in range(N_ACTIONS):
        if (mask >> a) & 1:
            q = deep[a]
            for f in range(N_FEAT):
                q += W[u, hidx, two, a, f] * x[f]
            if q > best:
                best = q
                best_a = a
    return best_a


@njit(cache=True)
def _play_seat(shoe, st, Q, eps, tags, cp, u, das, dbl_min, max_hands, resplit_aces, hit_split_aces,
               surrender, c1, c2, h_sum, h_ace, h_n, h_c1, h_c2, h_bet, h_stat, tr_s, tr_a, tr_m, tr_c,
               tr_len, lin, W, mlp, mdims, x, tr_x):
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
            if lin:
                _features(shoe, st, 1, x)
                a = _choose_model(W, mlp, mdims, lin, u, hidx, two, mask, eps, x)
            else:
                a = _choose(Q, b, u, hidx, two, mask, eps)
            k = tr_len[h]
            for r in range(10):
                tr_x[h, k, r] = st[2 + r]
            tr_x[h, k, 10] = shoe.shape[0] - st[0] + 1
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
           c1, c2, nh, h_sum, h_ace, h_n, h_c1, h_c2, h_bet, h_stat, tr_s, tr_a, tr_m, tr_c, tr_len,
           lin, W, mlp, mdims, Wr, x, hand_R, tr_x):
    """One round where every seat plays (and learns into) the shared table. Fills
    rewards[seat] and hand_R[seat, hand], and returns the bucket before the deal: of the
    count, or in linear mode (lin) of the edge predicted by Wr from the composition."""
    if shoe.shape[0] > 0 and st[0] >= cut:
        _shuffle(shoe, st)
    if lin:
        _features(shoe, st, 0, x)
        edge = 0.0
        for f in range(N_FEAT):
            edge += Wr[f] * x[f]
        pre_b = min(max(int(np.floor((100.0 * edge - cp[3]) / cp[4])), 0), int(cp[5]) - 1)
    else:
        pre_b = _count_bucket(shoe, st, 0, cp)

    for s in range(n_seats):
        c1[s] = _draw(shoe, st, tags)
    up = _draw(shoe, st, tags)
    for s in range(n_seats):
        c2[s] = _draw(shoe, st, tags)
    hole = _draw(shoe, st, tags)
    st[1] -= tags[hole]
    st[1 + hole] -= 1
    dealer_bj = (up == 1 and hole == 10) or (up == 10 and hole == 1)

    # insurance: half-bet side bet paying 2:1 on a dealer blackjack, decided on the count
    ins = 0.0
    if insurance and up == 1:
        outcome = 1.0 if dealer_bj else -0.5
        if lin:  # exact for a known composition: insure when over a third of the unseen cards are tens
            tens = (shoe.shape[0] // 52) * 16 - st[11]
            if 3 * tens > shoe.shape[0] - st[0] + 1:
                ins = outcome
        else:
            bi = _count_bucket(shoe, st, 1, cp)
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
        st[1 + hole] += 1
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
                           h_c2[s], h_bet[s], h_stat[s], tr_s[s], tr_a[s], tr_m[s], tr_c[s], tr_len[s],
                           lin, W, mlp, mdims, x, tr_x[s])
        for h in range(nh[s]):
            if h_stat[s, h] == 0:
                any_live = True

    st[1] += tags[hole]
    st[1 + hole] += 1
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

    for s in range(n_seats):
        R = hand_R[s]
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
def _table(n_decks, penetration):
    shoe = _new_shoe(n_decks)
    st = np.zeros(ST_SIZE, np.int64)
    if n_decks > 0:
        _shuffle(shoe, st)
    return shoe, st, int(penetration * shoe.shape[0])


@njit(cache=True)
def _work():
    """Per-worker scratch arrays for _round."""
    S, H = MAX_SEATS, MAX_HANDS
    return (np.zeros(S), np.zeros(S, np.int64), np.zeros(S, np.int64), np.zeros(S, np.int64),
            np.zeros((S, H), np.int64), np.zeros((S, H), np.bool_), np.zeros((S, H), np.int64),
            np.zeros((S, H), np.int64), np.zeros((S, H), np.int64), np.ones((S, H)),
            np.zeros((S, H), np.int64), np.zeros((S, H, MAX_STEPS, 4), np.int64),
            np.zeros((S, H, MAX_STEPS), np.int64), np.zeros((S, H, MAX_STEPS), np.int64),
            np.zeros((S, H, MAX_STEPS), np.int64), np.zeros((S, H), np.int64), np.zeros(N_FEAT),
            np.zeros((S, H)), np.zeros((S, H, MAX_STEPS, N_FEAT), np.int64))


@njit(cache=True)
def _run_worker(Q, N, Qi, Ni, stats, n_rounds, seed, learn, eps, tags, cp, n_decks, penetration,
                h17, das, dbl_min, max_hands, resplit_aces, hit_split_aces, peek, surrender, insurance,
                bj_payout, n_seats, lin, W, mlp, mdims, Wr):
    np.random.seed(seed)
    shoe, st, cut = _table(n_decks, penetration)
    (rewards, c1, c2, nh, h_sum, h_ace, h_n, h_c1, h_c2, h_bet, h_stat, tr_s, tr_a, tr_m, tr_c,
     tr_len, x, hand_R, tr_x) = _work()
    for _ in range(n_rounds):
        b = _round(shoe, st, Q, N, Qi, Ni, learn, eps, tags, cp, cut, h17, das, dbl_min, max_hands,
                   resplit_aces, hit_split_aces, peek, surrender, insurance, bj_payout, n_seats, rewards,
                   c1, c2, nh, h_sum, h_ace, h_n, h_c1, h_c2, h_bet, h_stat, tr_s, tr_a, tr_m, tr_c, tr_len,
                   lin, W, mlp, mdims, Wr, x, hand_R, tr_x)
        for s in range(n_seats):
            stats[s, b, 0] += 1.0
            stats[s, b, 1] += rewards[s]
            stats[s, b, 2] += rewards[s] * rewards[s]


@njit(parallel=True, cache=True)
def run_parallel(Qs, Ns, Qis, Nis, stats, n_rounds, seed, learn, eps, tags, cp, n_decks, penetration,
                 h17, das, dbl_min, max_hands, resplit_aces, hit_split_aces, peek, surrender, insurance,
                 bj_payout, n_seats, lin, W, mlp, mdims, Wr):
    """One independent table per worker, each with its own shoe, RNG stream and Q copy.
    stats[w, seat, count bucket] = (rounds, sum of results, sum of squared results).
    lin: act with the linear weights W (and bucket rounds by the edge Wr predicts) instead of Q."""
    for w in prange(Qs.shape[0]):
        _run_worker(Qs[w], Ns[w], Qis[w], Nis[w], stats[w], n_rounds, seed + 7919 * w, learn, eps,
                    tags, cp, n_decks, penetration, h17, das, dbl_min, max_hands, resplit_aces,
                    hit_split_aces, peek, surrender, insurance, bj_payout, n_seats, lin, W, mlp, mdims, Wr)


@njit(cache=True)
def _collect_worker(Q, Qi, n_rounds, seed, eps, tags, cp, n_decks, penetration, h17, das, dbl_min,
                    max_hands, resplit_aces, hit_split_aces, peek, surrender, insurance, bj_payout,
                    n_seats, lin, W, mlp, mdims, Wr, X, U, HI, T, M, A, N1, R1, N2, R2, P, PR, counts):
    """Play rounds and export every decision as a row: seen cards by rank and cards unseen (X),
    up card, hand index, two-card flag, legal-action mask, action; then the next decision of
    the same hand (N1, a row index) or its result (R1), and for a split the first decision
    of the new hand (N2) or its result (R2); N2 = -2 when the action is not a split.
    Also one row per round: the composition before the deal (P) and the mean result (PR)."""
    np.random.seed(seed)
    shoe, st, cut = _table(n_decks, penetration)
    (rewards, c1, c2, nh, h_sum, h_ace, h_n, h_c1, h_c2, h_bet, h_stat, tr_s, tr_a, tr_m, tr_c,
     tr_len, x, hand_R, tr_x) = _work()
    N = np.zeros((1, 1, 1, 1, 1), np.int64)
    Ni = np.zeros(1, np.int64)
    offs = np.zeros(MAX_HANDS, np.int64)
    n = 0
    rounds = 0
    room = n_seats * MAX_HANDS * MAX_STEPS
    while rounds < n_rounds and n + room <= X.shape[0]:
        if st[0] >= cut:
            _shuffle(shoe, st)
        _features(shoe, st, 0, x)
        for f in range(N_FEAT):
            P[rounds, f] = x[f]
        _round(shoe, st, Q, N, Qi, Ni, False, eps, tags, cp, cut, h17, das, dbl_min, max_hands,
               resplit_aces, hit_split_aces, peek, surrender, insurance, bj_payout, n_seats, rewards,
               c1, c2, nh, h_sum, h_ace, h_n, h_c1, h_c2, h_bet, h_stat, tr_s, tr_a, tr_m, tr_c, tr_len,
               lin, W, mlp, mdims, Wr, x, hand_R, tr_x)
        mean = 0.0
        for s in range(n_seats):
            mean += rewards[s] / n_seats
            total = 0
            for h in range(nh[s]):
                offs[h] = total
                total += tr_len[s, h]
            for h in range(nh[s]):
                L = tr_len[s, h]
                for k in range(L):
                    i = n + offs[h] + k
                    for f in range(N_FEAT):
                        X[i, f] = tr_x[s, h, k, f]
                    U[i] = tr_s[s, h, k, 1]
                    HI[i] = tr_s[s, h, k, 2]
                    T[i] = tr_s[s, h, k, 3]
                    M[i] = tr_m[s, h, k]
                    A[i] = tr_a[s, h, k]
                    if k + 1 < L:
                        N1[i] = n + offs[h] + k + 1
                        R1[i] = 0.0
                    else:
                        N1[i] = -1
                        R1[i] = hand_R[s, h]
                    N2[i] = -2
                    R2[i] = 0.0
                    if A[i] == SPLIT:
                        j = tr_c[s, h, k]
                        if tr_len[s, j] > 0:
                            N2[i] = n + offs[j]
                        else:
                            N2[i] = -1
                            R2[i] = hand_R[s, j]
            n += total
        PR[rounds] = mean
        rounds += 1
    counts[0] = n
    counts[1] = rounds


@njit(parallel=True, cache=True)
def collect_parallel(Q, Qi, n_rounds, seed, eps, tags, cp, n_decks, penetration, h17, das, dbl_min,
                     max_hands, resplit_aces, hit_split_aces, peek, surrender, insurance, bj_payout,
                     n_seats, lin, W, mlp, mdims, Wr, X, U, HI, T, M, A, N1, R1, N2, R2, P, PR, counts):
    """_collect_worker on independent tables; every output has a leading worker axis."""
    for w in prange(X.shape[0]):
        _collect_worker(Q, Qi, n_rounds, seed + 7919 * w, eps, tags, cp, n_decks, penetration, h17,
                        das, dbl_min, max_hands, resplit_aces, hit_split_aces, peek, surrender,
                        insurance, bj_payout, n_seats, lin, W, mlp, mdims, Wr, X[w], U[w], HI[w], T[w], M[w], A[w],
                        N1[w], R1[w], N2[w], R2[w], P[w], PR[w], counts[w])
