from typing import NamedTuple

import numpy as np


class ShoeView(NamedTuple):
    """What a player at the table can know about the shoe."""
    seen: tuple  # face-up cards dealt since the shuffle, by rank A, 2, ..., 9, T
    cards_left: int


class Deck:
    """Finite shoe dealt without replacement, reshuffled at the cut card. Counting is left
    to the player."""
    def __init__(self, deck_count = 6, penetration = 0.75):
        self._deck_count = deck_count
        self._stack = np.vectorize( lambda i: i+1 if i<10 else 10)(np.arange(0, 52 * self._deck_count) % 13)
        self._cut = int(penetration * 52 * deck_count)
        self._collected = []
        self._seen = np.zeros(11, int)
        self._dealt = 0
        self.shuffle()
    def shuffle(self):
        new_stack = np.concatenate([self._stack, np.array(self._collected, int)])
        self._collected.clear()
        self._stack = np.random.choice(new_stack, size=len(new_stack), replace=False)
        self._seen[:] = 0
        self._dealt = 0
    def need2shuffle(self):
        return len(self._stack) == 0
    def past_cut_card(self):
        return self._dealt >= self._cut
    def deal(self, face_up = True):
        if self.need2shuffle():
            self.shuffle()
        card = int(self._stack[0])
        self._stack  = self._stack[1:]
        self._dealt += 1
        if face_up:
            self._seen[card] += 1
        return card
    def reveal(self, card):
        self._seen[card] += 1
    def view(self) -> ShoeView:
        return ShoeView(tuple(int(n) for n in self._seen[1:]), len(self._stack))
    def collect(self, cards : list):
        self._collected.extend(cards)
        cards.clear()
    def all_card_collected(self) -> bool:
        return len(self._collected) + len(self._stack) == self._deck_count * 52
