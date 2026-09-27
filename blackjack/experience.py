"""Experience for learning from the shoe composition: every decision played by an agent,
exported as rows (see engine._collect_worker), and the features the models use."""
import os

import numpy as np

from blackjack import engine as E
from blackjack.counts import Bucketing, engine_params
from blackjack.qlearn import QTable

RANK_PER_DECK = np.array([4.0] * 9 + [16.0])


def features(X, n_decks):
    """Rows of seen counts by rank + cards unseen -> [1, excess of each rank left per deck]."""
    X = np.asarray(X, np.float64)
    left = n_decks * RANK_PER_DECK - X[:, :10]
    return np.hstack([np.ones((len(X), 1)), 52.0 * left / X[:, 10:11] - RANK_PER_DECK])


def collect(rules, n_rounds, seed=0, eps=0.3, qt: QTable = None, W=None, Wr=None, buckets=None,
            workers=None, mlp=None, mdims=None):
    """Play n_rounds and return the decisions as a dict of arrays (row links are global).
    The agent is the count table qt (with exploration eps), or the model W (plus the network
    part mlp, mdims when given), in which case rounds are bucketed by the edge Wr predicts."""
    workers = workers or os.cpu_count() or 1
    per = max(1, n_rounds // workers)
    lin = 0 if W is None else (2 if mlp is not None else 1)
    if mlp is None:
        mlp, mdims = np.zeros(1), np.zeros(3, np.int64)
    if lin:
        from blackjack.counts import SYSTEMS
        count, buckets = SYSTEMS["hi_lo"], buckets or Bucketing.around(0.0, 0.5, 8)
        Q, Qi = np.zeros((1, 1, 1, 1, E.N_ACTIONS)), np.zeros(buckets.n)
    else:
        count, buckets, Q, Qi = qt.count, qt.buckets, np.nan_to_num(qt.policy_Q(), nan=-np.inf), qt.Qi
        W, Wr = np.zeros((1, 1, 1, 1, E.N_FEAT)), np.zeros(E.N_FEAT)
    tags, cp = engine_params(count, buckets, rules.n_decks)
    cap = int(per * rules.n_seats * 2.5) + rules.n_seats * E.MAX_HANDS * E.MAX_STEPS
    z = lambda shape, dt: np.zeros((workers,) + shape, dt)
    X, U, HI, T, M, A = (z((cap, E.N_FEAT), np.int32), z((cap,), np.int8), z((cap,), np.int8),
                         z((cap,), np.int8), z((cap,), np.int8), z((cap,), np.int8))
    N1, R1, N2, R2 = z((cap,), np.int64), z((cap,), np.float32), z((cap,), np.int64), z((cap,), np.float32)
    P, PR, counts = z((per, E.N_FEAT), np.float32), z((per,), np.float32), z((2,), np.int64)
    E.collect_parallel(Q, Qi, per, seed, eps, tags, cp, rules.n_decks, rules.penetration, rules.h17,
                       rules.das, rules.double_min, rules.max_hands, rules.resplit_aces,
                       rules.hit_split_aces, rules.peek, rules.surrender, rules.insurance, rules.bj_payout,
                       rules.n_seats, lin, np.ascontiguousarray(W, np.float64), np.asarray(mlp, np.float64),
                       np.asarray(mdims, np.int64), np.asarray(Wr, np.float64),
                       X, U, HI, T, M, A, N1, R1, N2, R2, P, PR, counts)
    parts = {k: [] for k in ("X", "U", "H", "T", "M", "A", "N1", "R1", "N2", "R2", "P", "PR")}
    offset = 0
    for w in range(workers):
        n, r = counts[w]
        for k, arr in zip(("X", "U", "H", "T", "M", "A", "R1", "R2"), (X, U, HI, T, M, A, R1, R2)):
            parts[k].append(arr[w, :n])
        for k, arr in (("N1", N1), ("N2", N2)):
            links = arr[w, :n].copy()
            links[links >= 0] += offset
            parts[k].append(links)
        parts["P"].append(P[w, :r])
        parts["PR"].append(PR[w, :r])
        offset += n
    return {k: np.concatenate(v) for k, v in parts.items()}
