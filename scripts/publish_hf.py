"""Publish the trained tables and the results pages to Hugging Face.

    hf auth login                      # once, with a write token
    python scripts/publish_hf.py       # every game in configs/ with results

Model repo <user>/blackjack-counting: per game, the trained tables (q_<system>.npz), the
config and the results CSVs, with a model card. Space <user>/blackjack-counting-results:
a static site with each game's results page.
"""
import argparse
import glob
import html
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from blackjack import report
from blackjack.config import Config
from blackjack.counts import SYSTEMS

GITHUB = "https://github.com/apicquot/blackjack"

MODEL_CARD = """---
license: mit
library_name: numpy
tags:
- reinforcement-learning
- q-learning
- tabular
- blackjack
- card-counting
---

# Blackjack card counting: learned strategy tables

Tabular Q-learning agents that learned blackjack strategy **at every card count**,
for {n_games} Las Vegas games and {n_systems} counting systems ({systems}). No basic strategy or index table was given: every play,
index play and insurance decision was learned by self-play in a simulated shoe, with
every player at the table counting.

Code, method and validation: [{github}]({github}).
Interactive results: [{space}]({space}).

## Results at a glance

Best count per game. Money is in units won per 100 rounds dealt (1 unit = the minimum
bet), except the last column: units won per 100 hands actually played, since 0–3 sits
out the rounds without a player edge. Betting: 3 units at counts with a player edge,
otherwise 1 (1–3) or nothing (0–3).

{games_table}

## Files

For each game, a folder `<game>/` with:

- `config.yaml`: the rules (decks, penetration, soft 17, doubling, splits, surrender,
  payout, players)
- `q_<system>.npz`: the trained table of one count system
- `dqn_<model>.npz`: the benchmark agents without a predefined count, DQNs on the full shoe
  composition: `dqn_linear` (linear model) and, for single deck, `dqn_2x64` (plus a network
  with two hidden layers of 64). `W`: advantage weights per decision, `mlp`: the network
  part, `Wr`: predicted edge before the deal; load with `blackjack.dqn.ModelAgent.load`
- `summary.csv`, `by_seat.csv`: the results by count system and by seat

Each `.npz` holds:

| array | shape | meaning |
|---|---|---|
| `Q` | (count bucket, dealer up − 1, hand, two cards, action) | expected gain in initial bets of each action, then optimal play |
| `N` | same | visits; 0 means the action was never taken (illegal or unreached) |
| `Qi`, `Ni` | (count bucket,) | value of insurance, and visits |
| `buckets` | (lo, width, n) | count bucket k covers [lo + k·width, lo + (k+1)·width); the ends are open |
| `count` | | counting system; `rule_*` fields: the rules |

- **Hands:** index 0–17 is hard 4–21, 18–27 is soft 12–21, and 28–37 are splittable
  pairs A,A, 2,2 … 10,10.
- **Actions:** stand, hit, double, split, surrender.
- **Advantage:** `Q − max Q`, what each action costs against the best one.

## Use

```python
from huggingface_hub import hf_hub_download
from blackjack.qlearn import QTable        # pip install git+{github}

qt = QTable.load(hf_hub_download("{model}", "6deck_h17_das_ls/q_hi_lo.npz"))
qt.state(count=2, up=10, hand="H16")       # Q, visits and advantage of each action
print(qt.chart(count=2))                   # strategy chart at true count +2
```
"""

INDEX = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blackjack Counting Strategies</title>
<style>
body {{ font: 16px/1.55 system-ui, "Segoe UI", sans-serif; margin: 0; padding-block: 40px; padding-inline: 20px;
       background: #f4f6f2; color: #101612; }}
main {{ max-width: 880px; margin: 0 auto; }}
h1 {{ font-size: 34px; margin: 0 0 8px; }}
p {{ max-width: 66ch; color: #4a544e; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 20px; font-size: 14px; }}
th, td {{ padding: 9px 10px; border-bottom: 1px solid #e0e4dd; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
th {{ font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: #7d867f; }}
td {{ font-variant-numeric: tabular-nums; }}
a {{ color: #1f6a4c; }}
@media (prefers-color-scheme: dark) {{
  body {{ background: #0b0f0d; color: #eef2ee; }} p {{ color: #bcc5be; }}
  th, td {{ border-color: #252c27; }} a {{ color: #5dbb8f; }}
}}
</style></head>
<body><main>
<h1>Blackjack Counting Strategies</h1>
<p>Card-counting agents trained by self-play on popular Las Vegas games. Each game's page has the
strategy at every count, the value of each decision, the player's edge before the deal and the
return of each betting strategy. Figures are units won per 100 rounds (1 unit = the minimum bet).
Code: <a href="{github}">{github}</a>. Tables: <a href="{model_url}">{model}</a>.</p>
<table><thead><tr><th>Game</th><th>Best count</th><th>No count, flat bet</th><th>Best count, flat bet</th>
<th>Bet 1–3, play every round</th><th>Bet 0–3, sit out bad counts</th></tr></thead>
<tbody>{rows}</tbody></table>
</main></body></html>
"""

SPACE_CARD = """---
title: Blackjack Counting Results
emoji: 🃏
colorFrom: green
colorTo: gray
sdk: static
pinned: false
license: mit
---

Results pages of the card-counting agents in [{model}](https://huggingface.co/{model}).
Code: {github}.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", metavar="DIR", help="write what would be uploaded to DIR, upload nothing")
    a = p.parse_args()
    api = HfApi()
    user = api.whoami()["name"]
    model, space = f"{user}/blackjack-counting", f"{user}/blackjack-counting-results"
    space_url = f"https://huggingface.co/spaces/{space}"
    pages_url = f"https://{user}-blackjack-counting-results.static.hf.space"
    configs = [c for c in (Config.load(f) for f in sorted(glob.glob("configs/*.yaml")))
               if (c.docs_dir / "summary.csv").exists()]
    if not configs:
        raise SystemExit("no results: run scripts/run.py first")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = a.dry_run or tmp
        m = Path(tmp) / "model"
        for cfg in configs:
            (m / cfg.name).mkdir(parents=True)
            shutil.copy(f"configs/{cfg.name}.yaml", m / cfg.name / "config.yaml")
            for f in ("summary.csv", "by_seat.csv"):
                shutil.copy(cfg.docs_dir / f, m / cfg.name / f)
            for system in cfg.systems:
                shutil.copy(cfg.results_dir / f"q_{system}.npz", m / cfg.name / f"q_{system}.npz")
            for model in cfg.dqn:
                path = cfg.results_dir / f"dqn_{model[0]}.npz"
                if path.exists():
                    shutil.copy(path, m / cfg.name / path.name)
        games = "\n".join(report._games_table(configs, lambda c: f"{pages_url}/{c.name}/index.html"))
        systems = list(dict.fromkeys(s for c in configs for s in c.systems))
        names = ", ".join(SYSTEMS[s].description.split(", ")[0] for s in systems)
        (m / "README.md").write_text(MODEL_CARD.format(n_games=len(configs), n_systems=len(systems), systems=names,
                                                       github=GITHUB, space=space_url, games_table=games,
                                                       model=model), encoding="utf-8")

        s = Path(tmp) / "space"
        rows = []
        for cfg in configs:
            (s / cfg.name).mkdir(parents=True)
            shutil.copy(cfg.docs_dir / "strategy_card.html", s / cfg.name / "index.html")
            cells = report._games_table([cfg])[2].split("|")[2:-1]
            rows.append(f'<tr><td><a href="{cfg.name}/index.html">{html.escape(cfg.title)}</a></td>'
                        + "".join(f"<td>{html.escape(c.strip())}</td>" for c in cells[:5]) + "</tr>")
        (s / "index.html").write_text(INDEX.format(github=GITHUB, model=model, model_url=f"https://huggingface.co/{model}",
                                                   rows="\n".join(rows)), encoding="utf-8")
        (s / "README.md").write_text(SPACE_CARD.format(model=model, github=GITHUB), encoding="utf-8")

        if a.dry_run:
            print(f"wrote {Path(tmp) / 'model'} and {Path(tmp) / 'space'}")
            return
        api.create_repo(model, repo_type="model", private=a.private, exist_ok=True)
        api.upload_folder(repo_id=model, repo_type="model", folder_path=m, commit_message="Update tables and results")
        api.create_repo(space, repo_type="space", space_sdk="static", private=a.private, exist_ok=True)
        api.upload_folder(repo_id=space, repo_type="space", folder_path=s, commit_message="Update results pages")
    print(f"model: https://huggingface.co/{model}\nspace: {space_url}")


if __name__ == "__main__":
    main()
