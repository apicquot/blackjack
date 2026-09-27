from blackjack.rules import RuleSet


class Rules:
    """Table rules for the shoe env, from the RuleSet shared with the engine."""
    def __init__(self, rule_set : RuleSet = RuleSet()):
        if rule_set.n_decks == 0:
            raise ValueError("the shoe env needs a finite number of decks")
        self.rule_set = rule_set

    def is_split_ace(self, hand) -> bool:
        return hand.is_split() and hand._cards[0] == 1

    def can_hit(self, hand) -> bool:
        return self.rule_set.hit_split_aces or not self.is_split_ace(hand)

    def can_double(self, hand) -> bool:
        r = self.rule_set
        if len(hand._cards) != 2 or (hand.is_split() and not r.das) or not self.can_hit(hand):
            return False
        return r.double_min == 0 or (not hand.is_soft() and r.double_min <= hand.score() <= 11)

    def can_split(self, hand) -> bool:
        r = self.rule_set
        return (len(hand._cards) == 2 and hand._cards[0] == hand._cards[1]
                and len(hand._family) < r.max_hands and (r.resplit_aces or not self.is_split_ace(hand)))

    def can_surrender(self, hand) -> bool:
        return self.rule_set.surrender and len(hand._cards) == 2 and not hand.is_split()

    def is_blackjack(self, hand) -> bool:
        """No blackjack is paid on a split hand."""
        return len(hand._cards) == 2 and hand.score() == 21 and not hand.is_split()

    def dealer_hits(self, dealer) -> bool:
        return dealer.score() < 17 or (self.rule_set.h17 and dealer.score() == 17 and dealer.is_soft())

    def payout(self, hand, dealer) -> float:
        """Result per unit of the hand's bet."""
        if hand._surrendered:
            return -0.5
        if hand.busted():
            return -1
        player_bj, dealer_bj = self.is_blackjack(hand), self.is_blackjack(dealer)
        if player_bj:
            return 0 if dealer_bj else self.rule_set.bj_payout
        if dealer_bj or (not dealer.busted() and dealer.score() > hand.score()):
            return -1
        if dealer.busted() or hand.score() > dealer.score():
            return 1
        return 0
