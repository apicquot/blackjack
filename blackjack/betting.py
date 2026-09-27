"""Betting strategies on top of a trained agent: the play stays the learned one, only the
bet follows the count before the deal.

  flat 1        bet 1 every round
  1-3 max EV    play every round: 3 units when the value before the deal is positive, else 1
  1-3 ramp      play every round: 1 up to +0.5% edge, 2 up to +1%, 3 above
  0-3 max EV    sit out when the value is not positive (back-counting), else bet 3
  0-3 ramp      sit out when the value is not positive, then 1, 2, 3 as in the 1-3 ramp

Max EV is exactly optimal for expected return: each count's contribution, P(count) * bet *
value(count), is linear in the bet. Bets are chosen on one evaluation run and scored on an
independent one, so noise around zero edge does not flatter the result.
"""
import numpy as np

from blackjack.counts import Bucketing
from blackjack.qlearn import evaluate


def ramp(edge, low, cuts=(0.005, 0.01), top=3):
    return np.where(edge <= 0, low, np.minimum(1.0 + np.searchsorted(cuts, edge), top))


STRATEGIES = {
    "flat 1": lambda e: np.ones_like(e),
    "1-3 max EV": lambda e: np.where(e > 0, 3.0, 1.0),
    "1-3 ramp": lambda e: ramp(e, 1.0),
    "0-3 max EV": lambda e: np.where(e > 0, 3.0, 0.0),
    "0-3 ramp": lambda e: ramp(e, 0.0),
}


def evaluate_betting(qt, n_rounds):
    pick = qt.evaluate(n_rounds, seed=333)
    score = qt.evaluate(n_rounds, seed=444)
    out = dict(labels=score.labels, share=score.share.tolist(), edge=score.mean.tolist(),
               edge_se=score.stderr.tolist(), strategies={})
    for name, strategy in STRATEGIES.items():
        bets = strategy(pick.mean)
        stats = score.bet_stats(bets)
        out["strategies"][name] = {**{k: (None if np.isinf(v) else v) for k, v in stats.items()},
                                   "bets": bets.tolist()}
    out["seats"] = [dict(flat=seat.overall, se=seat.overall_stderr,
                         won13=100 * seat.bet_stats(STRATEGIES["1-3 max EV"](pick.mean))["ev_round"],
                         won03=100 * seat.bet_stats(STRATEGIES["0-3 max EV"](pick.mean))["ev_round"])
                    for seat in score.seats]
    return out


def evaluate_basic_strategy(qt, n_rounds):
    """Count-blind play for this shoe (the learned table with all counts pooled)."""
    res = evaluate(qt.pooled_Q()[None], qt.rules, qt.count, Bucketing.single(), n_rounds, seed=555)
    return dict(ev=res.overall, se=res.overall_stderr)
