from dataclasses import dataclass


@dataclass(frozen=True)
class RuleSet:
    """Table rules, shared by the engine, the exact solver and the shoe env."""
    n_decks: int = 6            # 0 = infinite deck
    penetration: float = 0.75   # share of the shoe dealt before reshuffling
    h17: bool = False           # dealer hits soft 17
    das: bool = True            # double after split
    double: str = "any"         # doubling on: any two cards, 9-11 or 10-11 (hard totals)
    max_hands: int = 2          # hands after splitting: 2 = no resplit, up to 4
    resplit_aces: bool = False
    hit_split_aces: bool = False
    peek: bool = True           # False = no hole card
    surrender: bool = False     # late surrender
    insurance: bool = True      # offered when the dealer shows an ace
    bj_payout: float = 1.5
    n_seats: int = 1            # players at the table, all counting agents

    def __post_init__(self):
        if not 1 <= self.n_seats <= 7:
            raise ValueError("n_seats must be 1 to 7")
        if self.n_decks and not 0 < self.penetration <= 0.9:
            raise ValueError("penetration must be in (0, 0.9]")
        if not 2 <= self.max_hands <= 4:
            raise ValueError("max_hands must be 2, 3 or 4")
        if self.double not in ("any", "9-11", "10-11"):
            raise ValueError("double must be 'any', '9-11' or '10-11'")

    @property
    def double_min(self):
        """Lowest hard total that may be doubled, 0 for any two cards."""
        return {"any": 0, "9-11": 9, "10-11": 10}[self.double]
