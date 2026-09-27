"""Exact infinite-deck solution by dynamic programming, same rules and state layout as the engine."""
from functools import lru_cache

import numpy as np

from blackjack.engine import (DOUBLE, HIT, N_ACTIONS, N_HAND, N_UP, SPLIT, STAND,
                              SURRENDER, hand_index)
from blackjack.rules import RuleSet

P = {c: (4 / 13 if c == 10 else 1 / 13) for c in range(1, 11)}
# dealer outcomes: 17..21, bust, blackjack
BUST, BJ = 5, 6


def _total(hard_sum, has_ace):
    return hard_sum + 10 if has_ace and hard_sum + 10 <= 21 else hard_sum


class InfiniteDeckSolver:
    def __init__(self, rules: RuleSet = RuleSet()):
        self.rules = rules
        self._dealer = {up: self._dealer_dist(up) for up in range(1, 11)}

    def _dealer_from(self, hard_sum, has_ace):
        @lru_cache(maxsize=None)
        def rec(hs, ace):
            out = np.zeros(7)
            t = _total(hs, ace)
            soft = ace and hs + 10 <= 21
            if t > 21:
                out[BUST] = 1
            elif t > 17 or (t == 17 and not (self.rules.h17 and soft)):
                out[t - 17] = 1
            else:
                for c, p in P.items():
                    out += p * rec(hs + c, ace or c == 1)
            return out
        return rec(hard_sum, has_ace)

    def _dealer_dist(self, up):
        out = np.zeros(7)
        for hole, p in P.items():
            if {up, hole} == {1, 10}:
                out[BJ] += p
            else:
                out += p * self._dealer_from(up + hole, up == 1 or hole == 1)
        if self.rules.peek:
            out[BJ] = 0
            out /= out.sum()
        return out

    def p_dealer_bj(self, up):
        return P[10] if up == 1 else (P[1] if up == 10 else 0.0)

    def stand(self, total, up):
        if total > 21:
            return -1.0
        d = self._dealer[up]
        ev = d[BUST] - d[BJ]
        for k in range(5):
            ev += d[k] * np.sign(total - (17 + k))
        return ev

    @lru_cache(maxsize=None)
    def v_multi(self, hard_sum, has_ace, up):
        t = _total(hard_sum, has_ace)
        if t > 21:
            return -1.0
        if t == 21:
            return self.stand(21, up)
        return max(self.stand(t, up), self.hit(hard_sum, has_ace, up))

    def hit(self, hard_sum, has_ace, up):
        return sum(p * self.v_multi(hard_sum + c, has_ace or c == 1, up) for c, p in P.items())

    def double(self, hard_sum, has_ace, up):
        return 2 * sum(p * self.stand(_total(hard_sum + c, has_ace or c == 1), up)
                       for c, p in P.items())

    def can_double(self, hard_sum, has_ace):
        m = self.rules.double_min
        soft = has_ace and hard_sum + 10 <= 21
        return m == 0 or (not soft and m <= hard_sum <= 11)

    def two_card_value(self, hard_sum, has_ace, up, split_child):
        t = _total(hard_sum, has_ace)
        if t == 21:
            return self.stand(21, up)
        vals = [self.stand(t, up), self.hit(hard_sum, has_ace, up)]
        if (self.rules.das or not split_child) and self.can_double(hard_sum, has_ace):
            vals.append(self.double(hard_sum, has_ace, up))
        if self.rules.surrender and not split_child:
            vals.append(-0.5)
        return max(vals)

    def split(self, rank, up):
        child = 0.0
        for c, p in P.items():
            if rank == 1:
                child += p * self.stand(_total(1 + c, True), up)
            else:
                child += p * self.two_card_value(rank + c, rank == 1 or c == 1, up, True)
        return 2 * child

    def q_table(self):
        """Q[up - 1, hand_index, two_cards, action], NaN where illegal."""
        Q = np.full((N_UP, N_HAND, 2, N_ACTIONS), np.nan)
        for up in range(1, 11):
            u = up - 1
            for ace in (False, True):
                for hs in range(2 if ace else 4, 22):
                    t = _total(hs, ace)
                    h = hand_index(hs, ace, 0)
                    Q[u, h, :, STAND] = self.stand(t, up)
                    Q[u, h, :, HIT] = self.hit(hs, ace, up)
                    if self.can_double(hs, ace):
                        Q[u, h, 1, DOUBLE] = self.double(hs, ace, up)
                    if self.rules.surrender:
                        Q[u, h, 1, SURRENDER] = -0.5
            for r in range(1, 11):
                h = hand_index(0, False, r)
                base = hand_index(2 * r, r == 1, 0)
                Q[u, h, 1, :] = Q[u, base, 1, :]
                Q[u, h, 1, SPLIT] = self.split(r, up)
        return Q

    def round_ev(self):
        Q = self.q_table()
        legal = np.array([True, True, True, True, self.rules.surrender])
        ev = 0.0
        for up in range(1, 11):
            pbj = self.p_dealer_bj(up)
            for c1 in range(1, 11):
                for c2 in range(1, 11):
                    p = P[up] * P[c1] * P[c2]
                    if {c1, c2} == {1, 10}:
                        ev += p * (1 - pbj) * self.rules.bj_payout
                        continue
                    h = hand_index(0, False, c1) if c1 == c2 else hand_index(c1 + c2, c1 == 1 or c2 == 1, 0)
                    best = np.nanmax(np.where(legal, Q[up - 1, h, 1], np.nan))
                    ev += p * (-pbj + (1 - pbj) * best if self.rules.peek else best)
        return ev
