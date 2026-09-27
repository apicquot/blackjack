"""Tabular Q-learning on the engine, and tools to read the learned table.

Everything is in units of the initial bet:
    Q(s, a)   expected gain of taking action a in state s, then playing optimally
    V(s)      max_a Q(s, a): the value of the state under optimal play
    A(s, a)   Q(s, a) - V(s) <= 0, the advantage: what choosing a instead of the best action costs
The value of a round before the deal, as a function of the count, is measured by
playing the learned policy (QTable.evaluate); it is what bet sizing should follow.
"""
import os
from dataclasses import asdict, dataclass

import numpy as np

from blackjack import engine as E
from blackjack.counts import SYSTEMS, Bucketing, CountSystem, default_bucketing, engine_params
from blackjack.rules import RuleSet

HI_LO = SYSTEMS["hi_lo"]


def _workers(workers):
    return workers or os.cpu_count() or 1


def _run(rules, count, buckets, Qs, Ns, Qis, Nis, n_rounds, seed, learn, eps):
    """Returns stats[seat, count bucket] = (rounds, sum of results, sum of squares)."""
    tags, cp = engine_params(count, buckets, rules.n_decks)
    stats = np.zeros((Qs.shape[0], rules.n_seats, buckets.n, 3))
    E.run_parallel(Qs, Ns, Qis, Nis, stats, n_rounds, seed, learn, eps, tags, cp, rules.n_decks,
                   rules.penetration, rules.h17, rules.das, rules.double_min, rules.max_hands,
                   rules.resplit_aces, rules.hit_split_aces, rules.peek, rules.surrender,
                   rules.insurance, rules.bj_payout, rules.n_seats, 0, _NO_W, _NO_MLP, _NO_MDIMS, _NO_WR)
    return stats.sum(axis=0)


_NO_W = np.zeros((1, 1, 1, 1, E.N_FEAT))
_NO_WR = np.zeros(E.N_FEAT)
_NO_MLP = np.zeros(1)
_NO_MDIMS = np.zeros(3, np.int64)


def _copies(x, W):
    return np.broadcast_to(x, (W,) + x.shape).copy()


def _merge(Q, N, Qs, Ns):
    """Combine worker tables that all started from (Q, N): exact for 1/N averaging."""
    W = Qs.shape[0]
    new = Ns.sum(axis=0) - (W - 1) * N
    sums = (Ns * Qs).sum(axis=0) - (W - 1) * N * Q
    return np.divide(sums, new, out=np.zeros(Q.shape), where=new > 0), new


@dataclass
class EdgeResult:
    """Expected gain per round as a function of the count before the deal, over all
    seats; seats holds the same result for each seat (first to act first)."""
    labels: list
    rounds: np.ndarray
    mean: np.ndarray
    stderr: np.ndarray
    overall: float
    overall_stderr: float
    seats: list = None

    @classmethod
    def from_stats(cls, labels, s, seats=None):
        n, sm, sq = s[:, 0], s[:, 1], s[:, 2]
        mean = np.divide(sm, n, out=np.zeros_like(sm), where=n > 0)
        var = np.divide(sq, n, out=np.zeros_like(sq), where=n > 0) - mean ** 2
        se = np.sqrt(np.divide(var, n, out=np.zeros_like(var), where=n > 0))
        tot = n.sum()
        m = sm.sum() / tot
        return cls(labels, n.astype(np.int64), mean, se, m, float(np.sqrt((sq.sum() / tot - m * m) / tot)), seats)

    @property
    def share(self):
        return self.rounds / self.rounds.sum()

    def bet_stats(self, bets):
        """Bet bets[k] units in count bucket k (0 = sit the round out).

        ev_round  units won per round dealt, including rounds sat out
        ev_hand   units won per hand actually played
        ev_unit   units won per unit wagered
        played    share of rounds played
        avg_bet   average bet on played hands
        sd_round  standard deviation per round dealt
        n0        rounds dealt until the expected win equals one standard deviation
        """
        b = np.asarray(bets, float)
        sh = self.share
        ex2 = self.stderr ** 2 * self.rounds + self.mean ** 2
        ev = float((sh * b * self.mean).sum())
        played = float(sh[b > 0].sum())
        wager = float((sh * b).sum())
        sd = float(np.sqrt((sh * b ** 2 * ex2).sum() - ev ** 2))
        return dict(ev_round=ev, ev_hand=ev / played, ev_unit=ev / wager, played=played,
                    avg_bet=wager / played, sd_round=sd, n0=(sd / ev) ** 2 if ev > 0 else np.inf)

    def __str__(self):
        lines = [f"overall EV/round {self.overall:+.4%} +/- {self.overall_stderr:.4%}",
                 "  count    share      EV/round    s.e."]
        for t, sh, m, s in zip(self.labels, self.share, self.mean, self.stderr):
            if sh:
                lines.append(f"{t:>7}  {sh:7.2%}   {m:+9.3%}   {s:.3%}")
        return "\n".join(lines)


class QTable:
    def __init__(self, Q, N, rules: RuleSet, count: CountSystem = HI_LO, buckets: Bucketing = None,
                 Qi=None, Ni=None):
        self.Q, self.N, self.rules, self.count = Q, N, rules, count
        self.buckets = buckets or default_bucketing(count, rules.n_decks)
        assert Q.shape[0] == self.buckets.n
        self.Qi = np.zeros(self.buckets.n) if Qi is None else Qi
        self.Ni = np.zeros(self.buckets.n, np.int64) if Ni is None else Ni

    @classmethod
    def train(cls, rules=RuleSet(), n_rounds=100_000_000, count=HI_LO, buckets=None, eps=0.3,
              epochs=10, workers=None, seed=0, init=None, verbose=True):
        """Workers train independent tables, merged by visit count after every epoch."""
        if isinstance(count, str):
            count = SYSTEMS[count]
        buckets = buckets or (init.buckets if init else default_bucketing(count, rules.n_decks))
        W = _workers(workers)
        shape = (buckets.n, E.N_UP, E.N_HAND, 2, E.N_ACTIONS)
        Q = np.zeros(shape) if init is None else init.Q.copy()
        N = np.zeros(shape, np.int64) if init is None else init.N.copy()
        Qi = np.zeros(buckets.n) if init is None else init.Qi.copy()
        Ni = np.zeros(buckets.n, np.int64) if init is None else init.Ni.copy()
        per_worker = max(1, n_rounds // (epochs * W))
        for ep in range(epochs):
            Qs, Ns, Qis, Nis = _copies(Q, W), _copies(N, W), _copies(Qi, W), _copies(Ni, W)
            _run(rules, count, buckets, Qs, Ns, Qis, Nis, per_worker, seed + 1_000_003 * ep, True, eps)
            Q, N = _merge(Q, N, Qs, Ns)
            Qi, Ni = _merge(Qi, Ni, Qis, Nis)
            if verbose:
                print(f"epoch {ep + 1}/{epochs}: {(ep + 1) * per_worker * W:,} rounds", flush=True)
        return cls(Q, N, rules, count, buckets, Qi, Ni)

    def legal_mask(self):
        return self.N > 0

    def V(self):
        q = np.where(self.legal_mask(), self.Q, -np.inf)
        v = q.max(axis=-1)
        return np.where(np.isfinite(v), v, np.nan)

    def advantage(self):
        return np.where(self.legal_mask(), self.Q - self.V()[..., None], np.nan)

    def policy_Q(self):
        return np.where(self.legal_mask(), self.Q, np.nan)

    def pooled_Q(self):
        """Count-blind Q (basic strategy view): Q averaged over count buckets, weighted by how
        often the state occurs in each. Every action gets the same weights; weighting by
        action visits would favour actions the agent mostly takes at good counts."""
        w = self.N.sum(axis=-1, keepdims=True).astype(float)
        known = self.N > 0
        wa = np.where(known, w, 0.0)
        tot = wa.sum(axis=0)
        return np.where(tot > 0, (wa * np.where(known, self.Q, 0.0)).sum(axis=0) / np.maximum(tot, 1), np.nan)

    def evaluate(self, n_rounds=100_000_000, rules=None, workers=None, seed=12345):
        return evaluate(self.policy_Q(), rules or self.rules, self.count, self.buckets,
                        n_rounds, workers, seed, self.Qi)

    def insurance_index(self, min_visits=20_000):
        """Lowest count bucket from which insurance is taken, or None."""
        take = (self.Qi > 0) & (self.Ni >= min_visits)
        return self.buckets.label(int(np.argmax(take))) if take.any() else None

    def bucket(self, value):
        return self.buckets.index(value)

    def state(self, count, up, hand, two=True):
        """e.g. qt.state(count=2, up=10, hand="H16")"""
        idx = (self.bucket(count), up - 1, E.HAND_LABELS.index(hand), int(two))
        adv = self.advantage()[idx]
        return {E.ACTION_NAMES[a]: dict(Q=self.Q[idx + (a,)], N=int(self.N[idx + (a,)]),
                                         advantage=adv[a])
                for a in range(E.N_ACTIONS) if self.N[idx + (a,)] > 0}

    def chart(self, count=None, min_visits=1000):
        """Two-card strategy chart at a count; None gives the count-blind chart."""
        if count is None:
            return strategy_chart(self.pooled_Q(), self.N.sum(axis=0), min_visits)
        b = self.bucket(count)
        return strategy_chart(self.policy_Q()[b], self.N[b], min_visits)

    def deviations(self, min_visits=20_000, min_gap=0.002):
        """[(bucket, hand, up, count-blind action, action at this count, gain)]"""
        def best(q):
            return np.where(np.isnan(q), -np.inf, q).argmax(-1)
        base = best(self.pooled_Q()[:, :, 1])
        pol = best(self.policy_Q()[:, :, :, 1])
        out = []
        for b in range(self.buckets.n):
            for u, h in np.argwhere(pol[b] != base):
                q, n = self.Q[b, u, h, 1], self.N[b, u, h, 1]
                a, a0 = pol[b, u, h], base[u, h]
                if n[a] >= min_visits and n[a0] >= min_visits and q[a] - q[a0] >= min_gap:
                    out.append((self.buckets.label(b), E.HAND_LABELS[h], "A" if u == 0 else str(u + 1),
                                E.ACTION_NAMES[a0], E.ACTION_NAMES[a], float(q[a] - q[a0])))
        return out

    def save(self, path):
        np.savez_compressed(path, Q=self.Q, N=self.N, Qi=self.Qi, Ni=self.Ni, count=self.count.name,
                            buckets=np.array([self.buckets.lo, self.buckets.width, self.buckets.n]),
                            **{f"rule_{k}": v for k, v in asdict(self.rules).items()})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        rules = RuleSet(**{k[5:]: d[k].item() for k in d.files if k.startswith("rule_")})
        lo, width, n = d["buckets"]
        return cls(d["Q"], d["N"], rules, SYSTEMS[str(d["count"])], Bucketing(lo, width, int(n)),
                   d["Qi"] if "Qi" in d.files else None, d["Ni"] if "Ni" in d.files else None)


def evaluate(Q, rules=RuleSet(), count=HI_LO, buckets=None, n_rounds=100_000_000,
             workers=None, seed=12345, Qi=None):
    """Play greedily with respect to Q (NaN actions are never chosen); insure where Qi > 0."""
    buckets = buckets or Bucketing.single()
    W = _workers(workers)
    Qi = np.zeros(buckets.n) if Qi is None else Qi
    Qs, Qis = _copies(Q, W), _copies(Qi, W)
    s = _run(rules, count, buckets, Qs, np.zeros(Qs.shape, np.int64), Qis,
             np.zeros(Qis.shape, np.int64), max(1, n_rounds // W), seed, False, 0.0)
    labels = [buckets.label(k) for k in range(buckets.n)]
    seats = [EdgeResult.from_stats(labels, s[i]) for i in range(s.shape[0])]
    return EdgeResult.from_stats(labels, s.sum(axis=0), seats)


def _letter(q):
    best = int(np.nanargmax(q))
    if best == E.DOUBLE:
        return "Dh" if q[E.HIT] >= q[E.STAND] else "Ds"
    if best == E.SURRENDER:
        return "Rh" if q[E.HIT] >= q[E.STAND] else "Rs"
    return "SHDPR"[best]


def strategy_chart(Q, N=None, min_visits=0):
    ups = list(range(2, 11)) + [1]
    rows = [f"H{t}" for t in range(5, 21)] + [f"S{t}" for t in range(13, 21)] \
        + ["2,2", "3,3", "4,4", "5,5", "6,6", "7,7", "8,8", "9,9", "10,10", "A,A"]
    out = ["hand   " + " ".join(f"{('A' if u == 1 else u):>3}" for u in ups)]
    for r in rows:
        h = E.HAND_LABELS.index(r)
        cells = []
        for u in ups:
            q = Q[u - 1, h, 1]
            if np.all(np.isnan(q)) or (N is not None and N[u - 1, h, 1].sum() < min_visits):
                cells.append("  .")
            else:
                cells.append(f"{_letter(q):>3}")
        out.append(f"{r:<6} " + " ".join(cells))
    return "\n".join(out)
