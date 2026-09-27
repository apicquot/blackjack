"""Play the object-oriented shoe env with exact basic strategy for a game config.

    python scripts/simulate_shoe.py configs/2deck_h17_das.yaml --rounds 200000
"""
import argparse

from blackjack.config import Config
from blackjack.envs.shoe.player_strategy import TableStrategy
from blackjack.envs.shoe.rules import Rules
from blackjack.envs.shoe.table import Table


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--rounds", type=int, default=200_000)
    a = p.parse_args()
    rules = Rules(Config.load(a.config).rules)
    table = Table(rules, 1, TableStrategy(rules))
    for _ in range(a.rounds):
        table.play()
    player = table._players[0]
    print(f"EV per round: {player.cash / player.rounds:+.3%} over {player.rounds:,} rounds")
