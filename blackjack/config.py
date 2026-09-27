"""A game to study: rules, count systems and training budget, read from a YAML file."""
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from blackjack.counts import SYSTEMS, default_bucketing
from blackjack.rules import RuleSet


@dataclass(frozen=True)
class Config:
    name: str            # file stem, used for the results/ and docs/ folders
    title: str
    rules: RuleSet
    systems: tuple
    rounds: int          # training rounds per count system
    eval_rounds: int     # rounds per evaluation run
    bucket_half: int = 6        # count buckets on each side of the centre
    bucket_scale: float = 1.0   # wider count steps for games whose count swings more
    linear_rounds: int = 0      # training rounds of the agent without a predefined count (0 = none)

    @classmethod
    def load(cls, path):
        path = Path(path)
        d = yaml.safe_load(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(RuleSet)}
        unknown = set(d.get("rules", {})) - known
        if unknown:
            raise ValueError(f"{path.name}: unknown rules {sorted(unknown)}")
        systems = tuple(d.get("systems", SYSTEMS))
        for s in systems:
            if s not in SYSTEMS:
                raise ValueError(f"{path.name}: unknown count system {s!r}")
        buckets = d.get("count_buckets", {})
        return cls(name=path.stem, title=d["title"], rules=RuleSet(**d.get("rules", {})),
                   systems=systems, rounds=int(float(d.get("rounds", 1e9))),
                   eval_rounds=int(float(d.get("eval_rounds", 2e8))),
                   bucket_half=int(buckets.get("half", 6)), bucket_scale=float(buckets.get("width_scale", 1.0)),
                   linear_rounds=int(float(d.get("linear", {}).get("rounds", 0))))

    def buckets(self, system):
        return default_bucketing(SYSTEMS[system], self.rules.n_decks, self.bucket_half, self.bucket_scale)

    @property
    def results_dir(self):
        return Path("results") / self.name

    @property
    def docs_dir(self):
        return Path("docs") / self.name
