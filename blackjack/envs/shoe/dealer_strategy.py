from blackjack.envs.shoe.action import Action
from blackjack.envs.shoe.deck import ShoeView
from blackjack.envs.shoe.hand import Hand
from blackjack.envs.shoe.rules import Rules
from blackjack.envs.shoe.strategy import Strategy

class DealerStrategy(Strategy):
    def __init__(self, rules : Rules):
        self._rules = rules
    def action(self, hand : Hand, dealer_up : int = None, shoe : ShoeView = None) -> Action:
        return Action.hit if self._rules.dealer_hits(hand) else Action.stand
