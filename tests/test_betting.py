import numpy as np
import pytest

from blackjack.qlearn import EdgeResult


def make_edge():
    # two buckets, 3/4 and 1/4 of rounds; per-round EV -1% and +2%; E[r^2] = 1.3 in both
    rounds = np.array([3_000_000, 1_000_000])
    mean = np.array([-0.01, 0.02])
    var = 1.3 - mean ** 2
    return EdgeResult(["-", "+"], rounds, mean, np.sqrt(var / rounds), 0.0, 0.0)


def test_flat_bet():
    s = make_edge().bet_stats([1, 1])
    assert s["ev_round"] == pytest.approx(0.75 * -0.01 + 0.25 * 0.02)
    assert s["played"] == 1 and s["avg_bet"] == 1
    assert s["sd_round"] == pytest.approx(np.sqrt(1.3 - s["ev_round"] ** 2))


def test_wonging():
    s = make_edge().bet_stats([0, 3])
    assert s["played"] == pytest.approx(0.25)
    assert s["ev_round"] == pytest.approx(0.25 * 3 * 0.02)
    assert s["ev_hand"] == pytest.approx(3 * 0.02)
    assert s["ev_unit"] == pytest.approx(0.02)
    assert s["sd_round"] == pytest.approx(np.sqrt(0.25 * 9 * 1.3 - s["ev_round"] ** 2))
    assert s["n0"] == pytest.approx((s["sd_round"] / s["ev_round"]) ** 2)


def test_pooled_q_uses_state_weights():
    from blackjack.counts import Bucketing
    from blackjack.qlearn import QTable
    from blackjack.rules import RuleSet
    Q = np.zeros((2, 10, 38, 2, 5))
    N = np.zeros((2, 10, 38, 2, 5), np.int64)
    s = (0, 5, 1)                                   # one state
    Q[(0,) + s + (0,)], N[(0,) + s + (0,)] = -0.5, 100   # low count: stand, mostly
    Q[(0,) + s + (2,)], N[(0,) + s + (2,)] = -1.0, 1     #            double, rarely
    Q[(1,) + s + (0,)], N[(1,) + s + (0,)] = -0.4, 1     # high count: stand, rarely
    Q[(1,) + s + (2,)], N[(1,) + s + (2,)] = 0.2, 100    #             double, mostly
    qt = QTable(Q, N, RuleSet(), buckets=Bucketing(-1.0, 1.0, 2))
    pooled = qt.pooled_Q()[s]
    assert pooled[0] == pytest.approx(-0.45)        # both buckets weighted by state visits (101 each)
    assert pooled[2] == pytest.approx(-0.40)        # not ~+0.19 as action-visit weighting would give
