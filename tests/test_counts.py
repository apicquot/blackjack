import numpy as np
import pytest

from blackjack.counts import SYSTEMS, Bucketing, default_bucketing
from blackjack.qlearn import evaluate
from blackjack.rules import RuleSet
from blackjack.solver import InfiniteDeckSolver


def test_system_properties():
    for name, s in SYSTEMS.items():
        assert s.balanced == (name != "ko"), name
    assert SYSTEMS["ko"].per_deck == 4
    assert SYSTEMS["ko"].irc(6) == -20
    assert SYSTEMS["wong_halves"].scale == 2
    assert list(SYSTEMS["wong_halves"].int_tags()) == [0, -2, 1, 2, 2, 3, 2, 1, 0, -1, -2]


def test_bucketing():
    b = Bucketing.around(0.0, 1.0, 6)
    assert b.n == 13
    assert [b.index(v) for v in (-99, -0.51, -0.49, 0.49, 0.51, 99)] == [0, 5, 6, 6, 7, 12]
    assert b.label(0) == "<=-6" and b.label(6) == "0" and b.label(12) == ">=+6"


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_every_system_predicts_edge(name):
    rules = RuleSet(n_decks=6)
    count = SYSTEMS[name]
    buckets = default_bucketing(count, rules.n_decks, half=3)
    Q = np.broadcast_to(InfiniteDeckSolver(rules).q_table(), (buckets.n, 10, 38, 2, 5)).copy()
    res = evaluate(Q, rules, count, buckets, 10_000_000)
    assert res.mean[-1] > 0.005          # high count: player edge
    assert res.mean[0] < res.overall - 0.005
    assert np.corrcoef(np.arange(buckets.n), res.mean)[0, 1] > 0.9
