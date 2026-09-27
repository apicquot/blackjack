import numpy as np
import pytest

from blackjack.counts import SYSTEMS, Bucketing, default_bucketing
from blackjack.engine import HAND_LABELS, HIT, SPLIT, STAND, DOUBLE
from blackjack.qlearn import QTable, evaluate
from blackjack.rules import RuleSet
from blackjack.solver import InfiniteDeckSolver

INF = RuleSet(n_decks=0)


def q_at(Q, hand, up, two=1):
    return Q[up - 1, HAND_LABELS.index(hand), two]


def test_solver_basic_strategy_cells():
    Q = InfiniteDeckSolver(INF).q_table()
    assert np.nanargmax(q_at(Q, "H16", 10)) == HIT
    assert np.nanargmax(q_at(Q, "H12", 4)) == STAND
    assert np.nanargmax(q_at(Q, "H11", 6)) == DOUBLE
    assert np.nanargmax(q_at(Q, "8,8", 10)) == SPLIT
    assert np.nanargmax(q_at(Q, "A,A", 6)) == SPLIT
    assert np.nanargmax(q_at(Q, "S18", 9)) == HIT


@pytest.mark.parametrize("rules", [INF, RuleSet(n_decks=0, h17=True, surrender=True),
                                   RuleSet(n_decks=0, peek=False, das=False),
                                   RuleSet(n_decks=0, h17=True, das=False, double="10-11", bj_payout=1.2)])
def test_engine_matches_exact_ev(rules):
    s = InfiniteDeckSolver(rules)
    res = evaluate(s.q_table()[None], rules, n_rounds=20_000_000)
    assert abs(res.overall - s.round_ev()) < 4 * res.overall_stderr


def test_qlearning_recovers_exact_q():
    qt = QTable.train(INF, 40_000_000, epochs=2, verbose=False)
    Qdp = InfiniteDeckSolver(INF).q_table()
    m = (qt.N[0] > 100_000) & ~np.isnan(Qdp)
    z = np.abs(qt.Q[0] - Qdp)[m] / (1.2 / np.sqrt(qt.N[0][m]))
    assert m.sum() > 100
    assert z.max() < 5


def test_count_predicts_edge():
    rules = RuleSet(n_decks=6)
    buckets = Bucketing.around(0.0, 1.0, 4)
    Q = np.broadcast_to(InfiniteDeckSolver(rules).q_table(), (buckets.n, 10, 38, 2, 5)).copy()
    res = evaluate(Q, rules, SYSTEMS["hi_lo"], buckets, 20_000_000)
    assert res.mean[-1] > 0.01           # TC >= +4: player edge
    assert res.mean[0] < -0.02           # TC <= -4: house edge
    assert np.all(np.diff(res.mean) > 0)


def test_insurance_learned_from_the_count():
    rules = RuleSet(n_decks=6)
    qt = QTable.train(rules, 30_000_000, epochs=2, verbose=False)
    b = qt.buckets
    assert qt.Qi[b.index(0)] < 0            # neutral count: insurance loses (about -1/26)
    assert qt.Qi[b.index(6)] > 0            # rich in tens: insurance pays
    assert qt.insurance_index() in ("+2", "+3", "+4")


def test_insurance_never_taken_on_infinite_deck():
    qt = QTable.train(INF, 5_000_000, epochs=1, verbose=False)
    assert qt.Qi[0] == pytest.approx(-1 / 26, abs=0.01)
    assert qt.insurance_index() is None
