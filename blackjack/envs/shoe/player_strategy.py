import numpy as np

from blackjack.engine import hand_index
from blackjack.envs.shoe.action import Action
from blackjack.envs.shoe.deck import ShoeView
from blackjack.envs.shoe.hand import Hand
from blackjack.envs.shoe.rules import Rules
from blackjack.envs.shoe.strategy import Strategy
from blackjack.solver import InfiniteDeckSolver

class PlayerStrategy(Strategy):
    """Simple: double 9-11 against a weaker up card, otherwise hit below 17."""
    def __init__(self, rules : Rules):
        self._rules = rules
    def action(self, hand : Hand, dealer_up : int, shoe : ShoeView = None) -> Action:
        up = 11 if dealer_up == 1 else dealer_up
        if self._rules.can_double(hand) and hand.score() in (9, 10, 11) and hand.score() > up:
            return Action.double
        elif hand.score() < 17:
            return Action.hit
        else:
            return Action.stand


class TableStrategy(Strategy):
    """Plays the best legal action of a Q table laid out like the engine's,
    Q[dealer_up - 1, hand_index, two_cards, action]; by default exact infinite-deck basic strategy."""
    def __init__(self, rules : Rules, Q = None):
        self._rules = rules
        self._Q = InfiniteDeckSolver(rules.rule_set).q_table() if Q is None else Q
    def action(self, hand : Hand, dealer_up : int, shoe : ShoeView = None) -> Action:
        legal = [Action.stand] + ([Action.hit] if self._rules.can_hit(hand) else [])
        if self._rules.can_double(hand):
            legal.append(Action.double)
        if self._rules.can_split(hand):
            legal.append(Action.split)
        if self._rules.can_surrender(hand):
            legal.append(Action.surrender)
        pair = hand._cards[0] if Action.split in legal else 0
        q = self._Q[dealer_up - 1, hand_index(hand._sum, hand._soft, pair), int(len(hand._cards) == 2)]
        return max(legal, key=lambda a: -np.inf if np.isnan(q[a.value]) else q[a.value])

