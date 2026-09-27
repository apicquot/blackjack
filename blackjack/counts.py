"""Card counting systems. Balanced systems play on the true count, unbalanced ones (KO)
on the running count."""
from dataclasses import dataclass

import numpy as np

RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "T")


@dataclass(frozen=True)
class CountSystem:
    name: str
    tags: tuple  # A, 2, ..., 9, T
    description: str = ""

    @property
    def per_deck(self):
        return 4 * sum(self.tags[:9]) + 16 * self.tags[9]

    @property
    def balanced(self):
        return self.per_deck == 0

    @property
    def level(self):
        return max(abs(t) for t in self.tags)

    @property
    def scale(self):
        return 1 if all(float(t).is_integer() for t in self.tags) else 2

    def int_tags(self):
        return np.array([0] + [round(t * self.scale) for t in self.tags], dtype=np.int64)

    def irc(self, n_decks):
        """Initial running count. For unbalanced systems (KO convention) the count ends at
        one deck's imbalance (+4 for KO) once the whole shoe is dealt."""
        return 0.0 if self.balanced else -self.per_deck * (n_decks - 1)


class Counter:
    """Player-side count from the cards seen since the shuffle."""
    def __init__(self, system: CountSystem, n_decks: int):
        self.system = system
        self.n_decks = n_decks

    def running_count(self, seen):
        return self.system.irc(self.n_decks) + float(np.dot(self.system.tags, seen))

    def true_count(self, seen, cards_left):
        return self.running_count(seen) * 52 / max(cards_left, 1)

    def value(self, seen, cards_left):
        if self.system.balanced:
            return self.true_count(seen, cards_left)
        return self.running_count(seen)


SYSTEMS = {s.name: s for s in [
    CountSystem("hi_lo", (-1, 1, 1, 1, 1, 1, 0, 0, 0, -1), "Hi-Lo, level 1"),
    CountSystem("hi_opt_1", (0, 0, 1, 1, 1, 1, 0, 0, 0, -1), "Hi-Opt I, level 1, ace neutral"),
    CountSystem("hi_opt_2", (0, 1, 1, 2, 2, 1, 1, 0, 0, -2), "Hi-Opt II, level 2, ace neutral"),
    CountSystem("omega_2", (0, 1, 1, 2, 2, 2, 1, 0, -1, -2), "Omega II, level 2, ace neutral"),
    CountSystem("zen", (-1, 1, 1, 2, 2, 2, 1, 0, 0, -2), "Zen Count, level 2"),
    CountSystem("wong_halves", (-1, 0.5, 1, 1, 1.5, 1, 0.5, 0, -0.5, -1), "Wong Halves, level 3 (half points)"),
    CountSystem("ko", (-1, 1, 1, 1, 1, 1, 1, 0, 0, -1), "Knock-Out, level 1, unbalanced (running count)"),
]}


@dataclass(frozen=True)
class Bucketing:
    """Bucket k covers [lo + k*width, lo + (k+1)*width); the end buckets are open-ended."""
    lo: float
    width: float
    n: int

    @classmethod
    def single(cls):
        return cls(0.0, 1.0, 1)

    @classmethod
    def around(cls, center, width, half):
        return cls(center - (half + 0.5) * width, width, 2 * half + 1)

    def index(self, value):
        return int(min(max(np.floor((value - self.lo) / self.width), 0), self.n - 1))

    def centers(self):
        return self.lo + self.width * (np.arange(self.n) + 0.5)

    def label(self, k):
        c = self.centers()[k]
        txt = f"{c:+g}" if c else "0"
        if self.n > 1 and k == 0:
            return "<=" + txt
        if self.n > 1 and k == self.n - 1:
            return ">=" + txt
        return txt


def default_bucketing(system: CountSystem, n_decks: int, half: int = 6, scale: float = 1.0) -> Bucketing:
    """2 * half + 1 buckets: true count in steps of 1 (2 for level-2 systems), KO running
    count in steps of about half the decks; scale widens the steps for games whose count
    swings more (few decks)."""
    if n_decks == 0:
        return Bucketing.single()
    if system.balanced:
        return Bucketing.around(0.0, (2.0 if system.level == 2 else 1.0) * scale, half)
    return Bucketing.around(system.irc(n_decks) / 2, max(1, round(n_decks / 2)) * scale, half)


def engine_params(system: CountSystem, buckets: Bucketing, n_decks: int):
    cp = np.array([system.scale, 1.0 if system.balanced else 0.0, system.irc(n_decks),
                   buckets.lo, buckets.width, buckets.n], dtype=np.float64)
    return system.int_tags(), cp
