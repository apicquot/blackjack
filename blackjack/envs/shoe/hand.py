from blackjack.envs.shoe.deck import Deck


class Hand:
    def __init__(self, deck: Deck):
        self._deck : Deck = deck
        self._cards : list[int] = []
        self._split_hands : list[Hand] = []
        self.reset()

    def reset(self, bet : float = 0):
        """Start a round; the cards of the previous one must have been collected."""
        self._soft : bool = False
        self._sum = 0
        self._done : bool = False
        self._surrendered : bool = False
        self._split_count : int = 0
        self._family : list[Hand] = [self]  # every hand of the round, after splits
        self.bet : float = bet

    def _add(self, card):
        if card == 1:
            self._soft = True
        self._sum += card
        self._cards.append(card)

    def _recount(self):
        self._sum = sum(self._cards)
        self._soft = 1 in self._cards

    def hit(self, face_up = True):
        card = self._deck.deal(face_up)
        self._add(card)
        self._done = self.busted()
        return card

    def stand(self):
        self._done = True

    def surrender(self):
        self._surrendered = True
        self._done = True

    def double(self):
        self.hit()
        self.bet *= 2
        self._done = True

    def split(self):
        hand = Hand(self._deck)
        hand._add(self._cards.pop())
        self._recount()
        self._split_count += 1
        hand._split_count = self._split_count
        hand.bet = self.bet
        hand._family = self._family
        self._family.append(hand)
        self._split_hands.append(hand)

    def collect(self):
        self._deck.collect(self._cards)
        for hand in self._split_hands:
            hand.collect()
        self._split_hands.clear()
        self.reset()

    def score(self):
        if self._soft and self._sum + 10 <= 21:
            return self._sum + 10
        else:
            return self._sum

    def is_soft(self):
        return self._soft and self._sum + 10 <= 21

    def is_split(self):
        return self._split_count > 0

    def busted(self):
        return self._sum > 21
