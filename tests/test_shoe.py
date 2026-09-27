import numpy as np
import pytest

from blackjack.counts import SYSTEMS, Counter
from blackjack.envs.shoe.action import Action
from blackjack.envs.shoe.deck import Deck
from blackjack.envs.shoe.hand import Hand
from blackjack.envs.shoe.player import Player
from blackjack.envs.shoe.player_strategy import TableStrategy
from blackjack.envs.shoe.rules import Rules
from blackjack.envs.shoe.strategy import Strategy
from blackjack.envs.shoe.table import Table
from blackjack.qlearn import evaluate
from blackjack.rules import RuleSet
from blackjack.solver import InfiniteDeckSolver


def hand_of(*cards, deck=None):
    hand = Hand(deck or Deck(1))
    for c in cards:
        hand._add(c)
    return hand


def test_deck_returns_every_card():
    deck = Deck(1)
    hand = Hand(deck)
    for _ in range(1000):
        while hand.score() < 17:
            hand.hit()
        hand.collect()
    assert len(deck._stack) + len(deck._collected) == 52


def test_cut_card():
    deck = Deck(1, penetration=0.5)
    for _ in range(25):
        deck.deal()
    assert not deck.past_cut_card()
    deck.deal()
    assert deck.past_cut_card()


def test_deck_view_and_counter():
    deck = Deck(1)
    assert deck.view() == ((0,) * 10, 52)
    dealt = [deck.deal() for _ in range(20)]
    hole = deck.deal(face_up=False)
    view = deck.view()
    assert view.cards_left == 31 and sum(view.seen) == 20
    assert view.seen[9] == dealt.count(10) and view.seen[0] == dealt.count(1)
    hi_lo = Counter(SYSTEMS["hi_lo"], 1)
    tags = {1: -1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1}
    assert hi_lo.running_count(view.seen) == sum(tags[c] for c in dealt)
    assert hi_lo.true_count(*view) == hi_lo.running_count(view.seen) * 52 / 31
    deck.reveal(hole)
    for _ in range(31):
        deck.deal()
    assert hi_lo.running_count(deck.view().seen) == 0      # a full deck sums to 0
    assert Counter(SYSTEMS["ko"], 6).value((0,) * 10, 312) == -20


def test_resplit_limits():
    rules = Rules(RuleSet(max_hands=4, resplit_aces=False))
    hand = hand_of(8, 8)
    hand.split()
    hand._add(8)
    assert rules.can_split(hand)
    hand.split()
    hand._add(8)
    hand.split()
    hand._add(8)
    assert len(hand._family) == 4 and not rules.can_split(hand)
    aces = hand_of(1, 1)
    assert rules.can_split(aces)
    aces.split()
    aces._add(1)
    assert not rules.can_split(aces) and not rules.can_hit(aces)
    assert Rules(RuleSet(max_hands=4, resplit_aces=True)).can_split(aces)


def test_split_keeps_one_card_each():
    hand = hand_of(8, 8)
    hand.split()
    assert hand._cards == [8] and hand._split_hands[0]._cards == [8]
    assert hand.is_split() and hand._split_hands[0].is_split()


def test_pay_split_hands():
    deck = Deck(1)
    dealer = hand_of(10, 8, deck=deck)
    player = Player(deck, Rules(), None, 0)
    player.reset(bet=1)
    for c in (8, 8):
        player._add(c)
    player.split()
    player._add(3)
    player._split_hands[0]._add(9)
    player.pay(dealer)
    assert player.total_bets == 2 and player.cash == -2


def test_payouts():
    rules = Rules()
    assert rules.payout(hand_of(7, 7, 7), hand_of(1, 10)) == -1
    assert rules.payout(hand_of(1, 10), hand_of(1, 10)) == 0
    assert rules.payout(hand_of(1, 10), hand_of(7, 7, 7)) == 1.5
    assert Rules(RuleSet(bj_payout=1.2)).payout(hand_of(1, 10), hand_of(10, 9)) == 1.2
    split_21 = hand_of(1, 10)
    split_21._split_count = 1
    assert rules.payout(split_21, hand_of(10, 10)) == 1


def test_legal_actions():
    rules = Rules(RuleSet(das=False, double="10-11", surrender=True))
    assert rules.can_double(hand_of(6, 5)) and not rules.can_double(hand_of(5, 4))
    assert not rules.can_double(hand_of(1, 9))                  # soft 20
    split = hand_of(5, 5)
    split.split()
    split._add(5)
    assert not rules.can_double(split) and not rules.can_split(split) and not rules.can_surrender(split)
    assert rules.can_surrender(hand_of(10, 6))


def test_dealer_soft_17():
    assert Rules(RuleSet(h17=True)).dealer_hits(hand_of(1, 6))
    assert not Rules(RuleSet(h17=False)).dealer_hits(hand_of(1, 6))


def test_illegal_action_rejected():
    class AlwaysSplit(Strategy):
        def action(self, hand, dealer_up, shoe):
            return Action.split
    table = Table(Rules(RuleSet(n_decks=1)), 1, AlwaysSplit())
    with pytest.raises(ValueError):
        for _ in range(100):
            table.play()


def test_counting_strategy_sees_only_the_up_card_and_shoe():
    class Counting(Strategy):
        def __init__(self):
            self.counter = Counter(SYSTEMS["hi_lo"], 1)
            self.seen = []
        def action(self, hand, dealer_up, shoe):
            self.seen.append((dealer_up, self.counter.true_count(*shoe)))
            return Action.hit if hand.score() < 17 else Action.stand
        def insurance(self, hand, shoe):
            return self.counter.true_count(*shoe) >= 3

    strategy = Counting()
    table = Table(Rules(RuleSet(n_decks=1, penetration=0.6)), 1, strategy)
    for _ in range(200):
        table.play()
    assert all(1 <= up <= 10 for up, _ in strategy.seen)
    assert any(tc != 0 for _, tc in strategy.seen)


@pytest.mark.parametrize("rule_set", [
    RuleSet(n_decks=2, penetration=0.65, h17=True, surrender=True),
    RuleSet(n_decks=6, h17=True, surrender=True, max_hands=4, resplit_aces=True, bj_payout=1.2)])
def test_shoe_env_matches_engine(rule_set):
    """Same basic strategy, same rules: the object-oriented table and the numba engine agree."""
    np.random.seed(7)
    table = Table(Rules(rule_set), 1, TableStrategy(Rules(rule_set)))
    for _ in range(400_000):
        table.play()
    player = table._players[0]
    shoe_ev = player.cash / player.rounds
    engine = evaluate(InfiniteDeckSolver(rule_set).q_table()[None], rule_set, n_rounds=20_000_000)
    se = np.hypot(1.15 / np.sqrt(player.rounds), engine.overall_stderr)
    assert abs(shoe_ev - engine.overall) < 4 * se
