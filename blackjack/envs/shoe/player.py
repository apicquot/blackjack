from blackjack.envs.shoe.deck import Deck
from blackjack.envs.shoe.hand import Hand
from blackjack.envs.shoe.rules import Rules
from blackjack.envs.shoe.strategy import Strategy


class Player(Hand):
    def __init__(self, deck : Deck, rules: Rules, strategy : Strategy, position):
        super().__init__(deck)
        self.cash = 0
        self.total_bets = 0
        self.rounds = 0
        self._rules : Rules = rules
        self._strategy : Strategy = strategy
        self._position = position
    def insure(self, dealer_blackjack : bool):
        side_bet = self.bet / 2
        self.cash += 2 * side_bet if dealer_blackjack else -side_bet
    def pay(self, dealer : Hand):
        self.rounds += 1
        self._pay(self, dealer)
    def _pay(self, hand: Hand, dealer : Hand):
        self.total_bets += hand.bet
        self.cash += self._rules.payout(hand, dealer) * hand.bet
        for split_hand in hand._split_hands:
            self._pay(split_hand, dealer)
