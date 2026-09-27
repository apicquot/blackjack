from blackjack.envs.shoe.action import Action
from blackjack.envs.shoe.deck import ShoeView
from blackjack.envs.shoe.hand import Hand

class Strategy:
    def bet(self, shoe : ShoeView) -> float:
        """Units to bet on the next round; 0 sits it out."""
        return 1
    def insurance(self, hand : Hand, shoe : ShoeView) -> bool:
        return False
    def action(self, hand : Hand, dealer_up : int, shoe : ShoeView) -> Action:
        pass
