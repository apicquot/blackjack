import numpy as np
import pytest

torch = pytest.importorskip("torch")

from blackjack import engine as E
from blackjack.dqn import DuelingNet, ModelAgent, train
from blackjack.experience import collect, features
from blackjack.qlearn import evaluate
from blackjack.rules import RuleSet
from blackjack.solver import InfiniteDeckSolver

RULES = RuleSet(n_decks=2, penetration=0.65, h17=True, insurance=False, n_seats=2, max_hands=4)


def test_experience_rows_are_consistent():
    d = collect(RULES, 200_000, eps=0.3, W=np.zeros((E.N_UP, E.N_HAND, 2, E.N_ACTIONS, E.N_FEAT)),
                Wr=np.zeros(E.N_FEAT))
    n = len(d["A"])
    assert ((d["M"] >> d["A"]) & 1).all()
    assert (d["N1"] < n).all() and (d["N2"] < n).all()
    later = d["N1"] >= 0
    assert (d["N1"][later] > np.flatnonzero(later)).all()
    assert ((d["N2"] > -2) == (d["A"] == E.SPLIT)).all()
    assert np.abs(features(d["X"], RULES.n_decks)[:, 1:].mean(0)).max() < 0.1


def test_dueling_masks_illegal_actions():
    net = DuelingNet()
    with torch.no_grad():
        net.WA.normal_()
        net.WV.normal_()
    d = torch.tensor([0, 5, 700])
    x = torch.randn(3, E.N_FEAT)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 0, 1]]).bool()
    q, v, a = net(d, x, mask)
    assert torch.isinf(q[0, 2:]).all() and torch.isfinite(q[1]).all()
    assert torch.allclose((a * mask).sum(1), torch.zeros(3), atol=1e-5)   # advantages centred on legal actions


def test_intercept_only_model_plays_like_the_table():
    """Weights with only an intercept play exactly the table's decisions: same results."""
    Q = InfiniteDeckSolver(RULES).q_table()
    W = np.zeros((E.N_UP, E.N_HAND, 2, E.N_ACTIONS, E.N_FEAT))
    W[..., 0] = np.nan_to_num(Q, nan=-1e9)
    table = evaluate(Q[None], RULES, n_rounds=4_000_000, seed=5)
    linear = ModelAgent(W, np.zeros(E.N_FEAT), RULES).evaluate(4_000_000, seed=5)
    assert linear.overall == pytest.approx(table.overall, abs=1e-12)


def test_training_learns_something():
    from blackjack.qlearn import QTable
    qt = QTable.train(RULES, 2_000_000, count="hi_lo", epochs=1, verbose=False)
    net = train(RULES, qt, rounds=8_000_000, chunk_rounds=2_000_000, lr=1e-2, lr_end=1e-3,
                log=lambda m: None, device="cpu")
    agent = ModelAgent.from_net(net, RULES, n_rounds=1_000_000)
    basic = evaluate(InfiniteDeckSolver(RULES).q_table()[None], RULES, n_rounds=4_000_000, seed=9).overall
    assert agent.evaluate(4_000_000, seed=9).overall > basic    # a short run already beats basic strategy


@pytest.mark.parametrize("activation", ["relu", "silu"])
def test_engine_network_matches_pytorch(activation):
    from blackjack.dqn import engine_model
    torch.manual_seed(0)
    net = DuelingNet((16, 8), activation)
    with torch.no_grad():
        for p in net.deep.parameters():
            p.normal_(0, 0.5)
    W, mlp, mdims = engine_model(net)
    rng = np.random.default_rng(1)
    for _ in range(20):
        u, h, t = int(rng.integers(10)), int(rng.integers(38)), int(rng.integers(2))
        x = np.r_[1.0, rng.normal(0, 2, 10)]
        out = np.zeros(E.N_ACTIONS)
        E._deep_advantages(mlp, mdims, u, h, t, x, out)
        z = torch.cat([torch.nn.functional.one_hot(torch.tensor([u]), 10), torch.nn.functional.one_hot(torch.tensor([h]), 38),
                       torch.tensor([[t]]), torch.tensor(x[None, 1:])], 1).float()
        expected = net.deep(z)[0, 1:].detach().numpy()
        assert np.allclose(out, expected, atol=1e-5)
