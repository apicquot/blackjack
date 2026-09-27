# blackjack

A blackjack table simulator and a card-counting reinforcement-learning agent. For
a given game (decks, rules, payout), the agent learns the best play for every hand
at every count, including insurance. It also gives the value of every decision and
the expected return of a round before the deal under several betting strategies.

gymnasium's `Blackjack-v1` draws from an infinite deck, so the past never
matters. Here the cards come from a finite shoe dealt without replacement. The
cards already seen change what is left, and a counting agent can exploit that.

Each game is described by a config file in `configs/`. The ones included are the
popular Las Vegas games, per the
[Vegas Advantage 2026 surveys](https://vegasadvantage.com/las-vegas-blackjack/):

| config | game |
|---|---|
| `6deck_h17_das_ls` | the common shoe game: dealer hits soft 17, double after split, late surrender, resplit to 4 hands, 3:2 |
| `8deck_h17_das_ls` | the same with 8 decks |
| `6deck_s17_das_ls_rsa` | the high-limit shoe game: dealer stands on soft 17, resplit aces too |
| `2deck_h17_das` | double deck: hits soft 17, double after split, 3:2, no surrender, no resplit |
| `1deck_h17_das_65` | single deck: every Las Vegas game pays 6:5 since 2024; hits soft 17, double after split |

Penetration (75% for shoes, 65% for 2 decks, 60% for 1 deck) is an assumption; the
surveys don't report it. Tables with continuous shuffling machines, common at low
limits, can't be counted and aren't modeled.

## Results at a glance

Each counting system trained its own agent on each game, then was scored on fresh
rounds. The first table gives the best count per game. The per-game tables below
it compare all systems, sorted by the return when you can sit out bad counts.

<!-- recap:start -->
<!-- recap:end -->

Each game's results page (`docs/<game>/strategy_card.html`) has the details: the
value of a round before the deal at each count, every betting strategy, the
strategy card with value and advantage at every count, and the index plays. GitHub
shows HTML files as source. To view a page, download it and open it in a browser,
or enable GitHub Pages on the `docs/` folder.

## Setup

The project lives in its own virtual environment at `~/venvs/bj_env`
(Python 3.13), outside the repo so it's never committed.

```bash
# create it once
python -m venv ~/venvs/bj_env

# install the package in editable mode: `import blackjack` then works from any
# directory, and code edits apply without reinstalling
~/venvs/bj_env/Scripts/python -m pip install -e ".[dev]"
```

Activate it to use `python` / `pytest` directly:

| shell | command |
|---|---|
| PowerShell | `~\venvs\bj_env\Scripts\Activate.ps1` |
| Git Bash | `source ~/venvs/bj_env/Scripts/activate` |
| Linux / WSL2 | `source ~/venvs/bj_env/bin/activate` (create the venv inside WSL; the folder is `bin/`, not `Scripts/`) |

VS Code picks the interpreter up from `.vscode/settings.json`. Dependencies:

- **`pyproject.toml`** declares them: numpy, numba, pyyaml; gymnasium with `[gym]`;
  pytest and gymnasium with `[dev]`.
- **`requirements.txt`** pins the exact versions the project was tested with.
  `pip install -r requirements.txt` reproduces that environment. No GPU is needed.
Training is tabular and runs on the CPU with numba, using every core.

```bash
pytest                                        # ~2 min, mostly simulation checks
python scripts/run.py configs/*.yaml          # train, evaluate and report every game
python scripts/run.py configs/2deck_h17_das.yaml --steps report     # rebuild one report
python scripts/simulate_shoe.py configs/2deck_h17_das.yaml          # basic strategy in the shoe env
```

`run.py` trains one agent per count system, evaluates the betting strategies,
writes `docs/<game>/` and refreshes the recap above. A game takes 30–45 minutes
on 8 cores. Finished steps are skipped, so an interrupted run resumes; `--force`
redoes them.

To study another game, copy a config and edit it:

```yaml
title: 2 decks, H17, DAS, 3:2
rules:
  n_decks: 2
  penetration: 0.65     # share of the shoe dealt before the shuffle
  h17: true             # dealer hits soft 17
  das: true             # double after split
  double: any           # any, 9-11 or 10-11
  max_hands: 2          # resplit up to this many hands (2 = no resplit, up to 4)
  resplit_aces: false
  hit_split_aces: false
  peek: true            # false: no hole card (European)
  surrender: false      # late surrender
  insurance: true
  bj_payout: 1.5        # 1.2 for 6:5
  n_seats: 3            # players at the table, all counting agents
systems: [hi_lo, hi_opt_1, hi_opt_2, omega_2, zen, wong_halves, ko]
count_buckets:           # optional: wider count steps when the count swings more (few decks)
  width_scale: 1.5
rounds: 5e8             # training rounds per count system (each round trains every seat)
eval_rounds: 1e8        # rounds per evaluation run
```

## Layout

| path | what |
|---|---|
| `configs/` | one YAML file per game |
| `blackjack/rules.py` | `RuleSet`: the table rules, shared by every part of the project |
| `blackjack/engine.py` | fast table simulator + Q-learning kernel (numba, ~5M rounds/s on 8 cores) |
| `blackjack/qlearn.py` | `QTable`: train, evaluate, `V`, `advantage`, strategy chart, index plays, save/load |
| `blackjack/counts.py` | counting systems, and `Counter`, the player-side running and true count |
| `blackjack/solver.py` | exact infinite-deck dynamic programming, the ground truth for tests |
| `blackjack/betting.py` | betting strategies and their evaluation |
| `blackjack/config.py`, `report.py` | config loading; results page, CSV and README recap |
| `blackjack/envs/shoe/` | object-oriented table (Deck, Hand, Player, Table, Rules, strategies) |
| `blackjack/envs/shoe/gym_env.py` | `ShoeEnv`, a gymnasium interface to the table for other RL libraries |
| `scripts/` | `run.py` (the pipeline), `simulate_shoe.py` |
| `tests/` | solver, engine, learner, count systems, betting math, shoe env |
| `docs/<game>/` | generated results page and `summary.csv` |
| `results/<game>/` | trained tables and evaluations (not committed, recreated by `run.py`) |

**Two implementations of the same game.**

- **`blackjack/envs/shoe/`** is readable, object-oriented Python, at about 30,000
  rounds per second. It deals a hole card, checks for dealer blackjack, offers
  insurance and surrender, splits, resplits and doubles as the rules allow, lets
  players bet or sit out, and reshuffles at the cut card. A strategy sees only its hand, the dealer's up card and a
  `ShoeView`: the face-up cards seen since the shuffle and the cards left.
  Counting is the player's job:

  ```python
  from blackjack.counts import SYSTEMS, Counter
  counter = Counter(SYSTEMS["hi_lo"], n_decks=6)
  tc = counter.true_count(*shoe_view)
  ```
- **`blackjack/engine.py`** plays the same rules in numba, about 150 times faster,
  and is what the agent trains on.

A test plays exact basic strategy in both and checks that they agree.

**For other RL algorithms**, `ShoeEnv` wraps the table as a standard
[gymnasium](https://gymnasium.farama.org/) environment (`pip install -e ".[gym]"`).
One episode is one round:

- **Decisions:** the agent makes all of them, namely its bet (0 sits out),
  insurance, and every playing decision, split hands included.
- **Reward:** the round's result, in units.
- **Observation:** a flat vector with the phase, the hand, the dealer's up card,
  the cards seen per rank, the cards left, and the bet.
- **Legal actions:** `action_masks()`, the convention used by `MaskablePPO` in
  sb3-contrib.

```python
from blackjack.config import Config
from blackjack.envs.shoe.gym_env import ShoeEnv

env = ShoeEnv(Config.load("configs/2deck_h17_das.yaml").rules, bets=(0, 1, 2, 3))
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)   # an action allowed by env.action_masks()
```

A test checks that it gives exactly the same results as `Table.play()` with the same
decisions.

## The RL algorithm

**Problem.** At each decision the state is
`(count bucket, dealer up card, hand, first two cards?)`:

- **Count bucket:** the counting system's value, cut into 13 buckets. For
  balanced systems it's the true count in steps of 1 (2 for level-2 systems).
  KO uses its running count. The end buckets are open-ended.
- **Hand:** hard 4–21, soft 12–21, or a splittable pair.
- **Actions:** stand, hit, double, split, surrender, masked by what's legal.
  Resplits are allowed up to the game's hand limit.

The count covers every card seen since the shuffle, including other players'
cards and the dealer's hole card once it's turned over. That's about 10,000
states × 5 actions, small enough for an exact table.

**Self-play at a full table.** Every player at the table (`n_seats`) is a counting
agent, and all of them play and learn into one shared table.

- **One table for every seat:** given the count, the up card and the hand, the best
  play doesn't depend on where you sit.
- **What the seat changes is information:** later seats see the earlier players'
  cards before they decide, and that already shows in their count.
- **More data per round:** every round trains every seat, so 3 players give 3
  times the experience.
- **Reported by seat:** results are given for each seat as well as for the table.

**Learner: tabular Q-learning**, off-policy temporal-difference learning,
undiscounted:

```
target = reward                       if the action ended the hand (stand, double, surrender, bust)
       = max_a' Q(next state, a')     after a hit that didn't bust
       = max Q(hand 1) + max Q(hand 2)   for a split
Q(s,a) += (target − Q(s,a)) / N(s,a)     # 1/N step: a running average of targets
```

- **Exploration:** ε-greedy with ε = 0.3, so every action gets sampled, even bad
  ones. Q-learning is off-policy, so exploration doesn't bias the values.
- **Delayed updates:** a round's decisions are updated once the dealer has
  played and the round is paid.
- **Parallel training:** one worker per CPU core, each playing its own shoe.
  Their tables are merged, weighted by visit counts, after each epoch. That merge
  is exact for 1/N averaging.
- **Nothing given in advance:** the learner starts from an all-zero table. It
  gets no basic strategy and no index table.

**Insurance** is a side bet of half the stake, offered when the dealer shows an
ace; it pays 2:1 if the dealer has blackjack. It doesn't change how the hand is
played, so its value is learned in its own small table, one value per count
bucket. The side bet's outcome is known whether or not it's taken, so every ace
up card updates it.

**What comes out**

- `Q(s,a)`: expected gain, in initial bets, of action `a` in state `s`, followed by optimal play
- `V(s) = max_a Q(s,a)`
- `A(s,a) = Q(s,a) − V(s) ≤ 0`, the advantage: what taking `a` costs versus the best action
- `Qi(count)`: expected result of insurance; take it where it's positive
- **Value before the deal:** expected gain of a round at each count, measured by
  playing the learned policy (`QTable.evaluate()`). This is what betting should follow.

**Why tabular Q-learning and not deep RL.** The state space fits in a table
exactly, and the goal is a precise value for every state. Blackjack returns are
very noisy (standard deviation about 1.15 bets per hand), so that takes millions
of samples per state. A neural network would add approximation error and train
far slower.

**Validation.**

- **Against the exact solution:** on an infinite deck the problem can be solved
  exactly by dynamic programming (`solver.py`). There, the learned Q table matches
  the exact one within sampling error.
- **Across rule sets:** the simulator's expected return matches the exact value
  for several rule sets, including H17 with surrender, no hole card, and 6:5 with
  doubling on 10–11 only.
- **Resplits:** the exact solver doesn't cover them. Their effect, +0.06% for
  resplitting to 4 hands and +0.08% more for resplitting aces, is in line with
  published figures.
- **Insurance:** it's learned to be worth taking from about true count +3, the
  textbook Hi-Lo index, and never on an infinite deck.

## Betting strategies

The playing strategy stays the learned one; only the bet follows the count before
the deal.

- **Max EV:** bet the maximum (3) when the value before the deal is positive,
  otherwise the minimum: 1 if you must play every round, 0 if you can sit out
  (back-counting). This maximizes expected return exactly. Each count's
  contribution, P(count) · bet · value(count), is linear in the bet, so the best
  bet is always at one end of the range.
- **Ramp:** raise the bet in steps as the edge grows: 1 up to +0.5%, 2 up to +1%,
  3 above. It earns a little less and swings much less.

Bet levels are chosen from one evaluation run and scored on another, so noise
around zero edge can't flatter the result.

Columns:

- **EV per hand:** units won per round actually played.
- **per 100 rounds:** units won per 100 rounds dealt, including rounds sat out.
  This is the earning rate at the table.
- **per unit bet:** units won per unit wagered.
- **SD:** standard deviation per round dealt.
- **N0:** rounds dealt until the expected win equals one standard deviation.
  Lower means the edge shows through the swings sooner.

The **count-blind baseline** ("no count") is this shoe's basic strategy: the
learned table with all counts pooled.

## Limits and next steps

- Each agent sees the shoe only through its system's count, not the full
  remaining composition.
- No ace side count for the ace-neutral systems, and no Red Seven (the simulator
  has no suits).
- Betting assumes you can sit out freely and ignores table limits. A
  risk-adjusted (Kelly) bet, which maximizes EV minus a variance penalty per
  count, is a small extension of `EdgeResult.bet_stats`.
- Every player at the table uses the same strategy table, so the model has no weak
  players whose cards arrive in a different way; that effect on the count is small.

## License

MIT, see [LICENSE](LICENSE).
