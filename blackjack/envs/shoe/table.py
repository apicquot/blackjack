from typing import NamedTuple

from blackjack.envs.shoe.action import Action
from blackjack.envs.shoe.dealer_strategy import DealerStrategy
from blackjack.envs.shoe.deck import Deck
from blackjack.envs.shoe.hand import Hand
from blackjack.envs.shoe.player import Player
from blackjack.envs.shoe.player_strategy import PlayerStrategy
from blackjack.envs.shoe.rules import Rules
from blackjack.envs.shoe.strategy import Strategy


class Decision(NamedTuple):
    """A point in the round where a player must answer: kind is "bet", "insurance" or "play"."""
    kind: str
    player: Player
    hand: Hand
    up: int


class Table():
    def __init__(self, rules : Rules = None, player_count = 1, strategy : Strategy = None):
        self._rules = rules or Rules()
        r = self._rules.rule_set
        self._deck = Deck(r.n_decks, r.penetration)
        self._dealer = Hand(self._deck)
        self._dealer_strategy = DealerStrategy(self._rules)
        strategy = strategy or PlayerStrategy(self._rules)
        self._players = [Player(self._deck, self._rules, strategy, position) for position in range(player_count)]

    def play(self):
        """Play one round, asking each player's strategy for its decisions."""
        round_ = self.round()
        try:
            decision = next(round_)
            while True:
                decision = round_.send(self.answer(decision))
        except StopIteration:
            pass

    def answer(self, decision : Decision):
        strategy, view = decision.player._strategy, self._deck.view()
        if decision.kind == "bet":
            return strategy.bet(view)
        if decision.kind == "insurance":
            return strategy.insurance(decision.hand, view)
        return strategy.action(decision.hand, decision.up, view)

    def round(self):
        """One round as a generator: yields a Decision whenever a player must choose, and
        expects the answer (bet size, insure or not, Action) to be sent back."""
        r = self._rules.rule_set
        if self._deck.past_cut_card():
            self._deck.shuffle()

        for player in self._players:
            player.reset(bet=(yield Decision("bet", player, player, 0)))
        seated = [p for p in self._players if p.bet > 0]

        for player in seated:
            player.hit()
        up = self._dealer.hit()
        for player in seated:
            player.hit()
        hole = self._dealer.hit(face_up=False)
        dealer_bj = self._rules.is_blackjack(self._dealer)

        if r.insurance and up == 1:
            for player in seated:
                if (yield Decision("insurance", player, player, up)):
                    player.insure(dealer_bj)

        if not (r.peek and dealer_bj):
            for player in seated:
                if self._rules.is_blackjack(player):
                    player.stand()
                yield from self._play_hands(player, player, up)
        self._deck.reveal(hole)
        if not dealer_bj and any(self._live(p) for p in seated):
            while not self._dealer._done and self._dealer.score() < 21:
                self._apply(self._dealer, self._dealer_strategy.action(self._dealer))
        for player in seated:
            player.pay(self._dealer)
            player.collect()
        self._dealer.collect()

        if not self._deck.all_card_collected():
            raise Exception("some cards have been lost!!!!")

    def _play_hands(self, player : Player, hand : Hand, up : int):
        while not hand._done:
            if len(hand._cards) == 1:
                hand.hit()
            elif hand.score() == 21 or not (self._rules.can_hit(hand) or self._rules.can_split(hand)):
                hand.stand()
            else:
                self._apply(hand, (yield Decision("play", player, hand, up)))
        for split_hand in hand._split_hands:
            yield from self._play_hands(player, split_hand, up)

    def _apply(self, hand : Hand, a : Action):
        if a == Action.hit and self._rules.can_hit(hand):
            hand.hit()
        elif a == Action.stand:
            hand.stand()
        elif a == Action.double and self._rules.can_double(hand):
            hand.double()
        elif a == Action.split and self._rules.can_split(hand):
            hand.split()
        elif a == Action.surrender and self._rules.can_surrender(hand):
            hand.surrender()
        else:
            raise ValueError(f"illegal action {a} with {hand._cards}")

    def _live(self, hand : Hand):
        if not (hand.busted() or hand._surrendered or self._rules.is_blackjack(hand)):
            return True
        return any(self._live(h) for h in hand._split_hands)
