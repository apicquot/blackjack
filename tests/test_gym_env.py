import numpy as np

from blackjack.envs.shoe.action import Action
from blackjack.envs.shoe.gym_env import BET0, DECLINE, INSURE, ShoeEnv
from blackjack.envs.shoe.player_strategy import TableStrategy
from blackjack.envs.shoe.rules import Rules
from blackjack.envs.shoe.table import Table
from blackjack.rules import RuleSet

RULES = RuleSet(n_decks=2, penetration=0.65, h17=True, surrender=True, max_hands=4)


def test_random_masked_play():
    env = ShoeEnv(RuleSet(n_decks=2, h17=True, max_hands=4, n_seats=3), bets=(0, 1, 2, 3), seat=1)
    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=1)
    episodes = 0
    while episodes < 2000:
        assert env.observation_space.contains(obs)
        action = rng.choice(np.flatnonzero(env.action_masks()))
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            episodes += 1
            obs, info = env.reset()


def test_sitting_out_pays_nothing():
    env = ShoeEnv(RULES, bets=(0, 1))
    env.reset(seed=3)
    assert env.action_masks()[BET0:].all()
    _, reward, terminated, _, _ = env.step(BET0)
    assert terminated and reward == 0


def test_env_matches_table_play():
    """The gym interface and Table.play give identical results with the same decisions."""
    rules = Rules(RULES)
    np.random.seed(11)
    table = Table(rules, 1, TableStrategy(rules))
    for _ in range(5000):
        table.play()
    expected = table._players[0].cash

    np.random.seed(11)
    env = ShoeEnv(RULES)
    strategy = TableStrategy(rules)
    total, rounds = 0.0, 0
    _, info = env.reset()
    total += info["skipped_reward"]
    while True:
        d = env._decision
        if d.kind == "insurance":
            action = DECLINE if not strategy.insurance(d.hand, None) else INSURE
        else:
            action = strategy.action(d.hand, d.up).value
        _, reward, terminated, _, _ = env.step(action)
        if terminated:
            total += reward
            if env.agent.rounds >= 5000:
                break
            _, info = env.reset()
            total += info["skipped_reward"]
    assert env.agent.rounds == 5000
    assert np.isclose(total, expected)
