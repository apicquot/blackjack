"""Gymnasium interface to the shoe table, for any RL library.

One episode is one round, seen from the agent's seat. The agent answers every
decision of the round: its bet (when more than one bet size is allowed), insurance,
and each playing decision, split hands included. The reward, given at the end, is
the round's result in units.

Actions (Discrete):   0 stand, 1 hit, 2 double, 3 split, 4 surrender,
                      5 decline insurance, 6 insure, 7 + i  bet bets[i]
Observation (Box):    phase one-hot (bet, insurance, play), hand total, soft, splittable
                      pair rank, number of cards, split hand, dealer up card, face-up
                      cards seen since the shuffle by rank (10), cards left, bet
Legal actions:        env.action_masks() (the convention of sb3-contrib's MaskablePPO)

With a single bet size there is no bet decision; rounds that then need no decision at
all (a blackjack without an insurance offer, a dealer blackjack) are played through,
and their results are added to the next episode's info["skipped_reward"].
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from blackjack.envs.shoe.action import Action
from blackjack.envs.shoe.player_strategy import TableStrategy
from blackjack.envs.shoe.rules import Rules
from blackjack.envs.shoe.table import Table
from blackjack.rules import RuleSet

N_PLAY, DECLINE, INSURE, BET0 = 5, 5, 6, 7
OBS_SIZE = 3 + 5 + 1 + 10 + 1 + 1


class ShoeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, rule_set: RuleSet = RuleSet(), bets=(1,), seat=0, others=None):
        """The agent plays seat `seat` (0 = first to act) of rule_set.n_seats; the other
        players follow `others`, exact basic strategy by default."""
        rules = Rules(rule_set)
        self.bets = tuple(bets)
        self.table = Table(rules, rule_set.n_seats, others or TableStrategy(rules))
        self.agent = self.table._players[seat]
        self.action_space = spaces.Discrete(BET0 + len(self.bets))
        high = np.array([1, 1, 1, 31, 1, 10, 21, 1, 10] + [32 * rule_set.n_decks] * 10
                        + [52 * rule_set.n_decks, max(self.bets)], np.float32)
        self.observation_space = spaces.Box(0, high, dtype=np.float32)
        self._round = None
        self._decision = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        skipped = 0.0
        while True:
            cash = self.agent.cash
            self._round = self.table.round()
            if self._advance(self._send(None)):
                self._start_cash = cash
                return self._obs(), {"skipped_reward": skipped, "action_mask": self.action_masks()}
            skipped += self.agent.cash - cash

    def step(self, action):
        if not self.action_masks()[action]:
            raise ValueError(f"illegal action {action} in phase {self._decision.kind}")
        cash = self._start_cash
        if self._advance(self._send(self._answer(int(action)))):
            return self._obs(), 0.0, False, False, {"action_mask": self.action_masks()}
        return self._obs(), float(self.agent.cash - cash), True, False, {}

    def action_masks(self):
        mask = np.zeros(self.action_space.n, bool)
        d = self._decision
        if d is None:
            return mask
        if d.kind == "bet":
            mask[BET0:] = True
        elif d.kind == "insurance":
            mask[[DECLINE, INSURE]] = True
        else:
            rules, hand = self.table._rules, d.hand
            mask[Action.stand.value] = True
            mask[Action.hit.value] = rules.can_hit(hand)
            mask[Action.double.value] = rules.can_double(hand)
            mask[Action.split.value] = rules.can_split(hand)
            mask[Action.surrender.value] = rules.can_surrender(hand)
        return mask

    def _send(self, answer):
        try:
            return self._round.send(answer)
        except StopIteration:
            return None

    def _advance(self, decision):
        """Answer other players' decisions until the agent must choose; False at round end."""
        while decision is not None and (decision.player is not self.agent
                                        or (decision.kind == "bet" and len(self.bets) == 1)):
            answer = self.bets[0] if decision.player is self.agent else self.table.answer(decision)
            decision = self._send(answer)
        self._decision = decision
        return decision is not None

    def _answer(self, action):
        if self._decision.kind == "bet":
            return self.bets[action - BET0]
        if self._decision.kind == "insurance":
            return action == INSURE
        return Action(action)

    def _obs(self):
        obs = np.zeros(OBS_SIZE, np.float32)
        view = self.table._deck.view()
        obs[9:19] = view.seen
        obs[19] = view.cards_left
        d = self._decision
        if d is None:
            return obs
        obs[("bet", "insurance", "play").index(d.kind)] = 1
        if d.kind != "bet":
            obs[20] = d.hand.bet
            hand = d.hand
            obs[3] = hand.score()
            obs[4] = hand.is_soft()
            obs[5] = hand._cards[0] if self.table._rules.can_split(hand) else 0
            obs[6] = len(hand._cards)
            obs[7] = hand.is_split()
            obs[8] = d.up
        return obs
