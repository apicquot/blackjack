"""Results page, recap CSV and README recap for a game config."""
import csv
import json
from pathlib import Path

import numpy as np

from blackjack.counts import SYSTEMS
from blackjack.engine import HAND_LABELS
from blackjack.qlearn import QTable
from blackjack.solver import InfiniteDeckSolver

TEMPLATE = Path(__file__).with_name("report_template.html")
UPS = list(range(2, 11)) + [1]
ROWS_TWO = [f"H{t}" for t in range(5, 21)] + [f"S{t}" for t in range(13, 21)] \
    + ["2,2", "3,3", "4,4", "5,5", "6,6", "7,7", "8,8", "9,9", "10,10", "A,A"]
ROWS_MULTI = [f"H{t}" for t in range(12, 21)] + [f"S{t}" for t in range(13, 21)]
RECAP_START, RECAP_END = "<!-- recap:start -->", "<!-- recap:end -->"
RECAP_COLUMNS = ["system", "type", "flat_ev", "play_gain", "edge_rounds",
                 "units_per_100_bet_1_3", "units_per_100_bet_0_3", "ev_per_hand_bet_0_3"]


def _r4(x):
    return None if np.isnan(x) else round(float(x), 4)


def _cells(Q, N, rows, two):
    return [[{"q": [_r4(x) for x in Q[u - 1, HAND_LABELS.index(r), two]],
              "n": [int(x) for x in N[u - 1, HAND_LABELS.index(r), two]]} for u in UPS] for r in rows]


def _best_action(Q, rows, two):
    out = []
    for r in rows:
        row = []
        for u in UPS:
            q = Q[u - 1, HAND_LABELS.index(r), two]
            row.append(-1 if np.all(np.isnan(q)) else int(np.where(np.isnan(q), -np.inf, q).argmax()))
        out.append(row)
    return out


def index_plays(qt):
    """Per (hand, up, change): the count where the deviation from count-blind play starts."""
    labels = [qt.buckets.label(k) for k in range(qt.buckets.n)]
    centers = qt.buckets.centers()
    mode = int(np.argmax(qt.N.sum(axis=(1, 2, 3, 4))))
    first = {}
    for lbl, hand, up, a0, a1, _ in qt.deviations():
        b = labels.index(lbl)
        if b == mode:
            continue
        side = 1 if b > mode else -1
        key = (hand, up, a0, a1, side)
        if key not in first or (b < first[key] if side > 0 else b > first[key]):
            first[key] = b
    plays = [dict(hand=h, up=u, frm=a0, to=a1, side=sd, label=labels[b], center=float(centers[b]))
             for (h, u, a0, a1, sd), b in first.items()]
    plays.sort(key=lambda d: (d["side"], d["side"] * d["center"], d["hand"]))
    ins = qt.insurance_index()
    if ins:
        k = labels.index(ins)
        plays.insert(0, dict(hand="Insurance", up="A", frm="decline", to="insure", side=1, label=ins,
                             center=float(centers[k])))
    return plays, mode


def build(cfg):
    """Write docs/<game>/strategy_card.html and docs/<game>/summary.csv from results/<game>/."""
    basic = json.loads((cfg.results_dir / "basic_strategy.json").read_text())["ev"]
    systems, summary, betting = {}, [], {}
    for name in cfg.systems:
        qt = QTable.load(cfg.results_dir / f"q_{name}.npz")
        bet = json.loads((cfg.results_dir / f"betting_{name}.json").read_text())
        betting[name] = dict(labels=bet["labels"], strategies=bet["strategies"])
        Qp, pooled = qt.policy_Q(), qt.pooled_Q()
        plays, mode = index_plays(qt)
        share, edge, se = map(np.array, (bet["share"], bet["edge"], bet["edge_se"]))
        flat = bet["strategies"]["flat 1"]
        systems[name] = dict(
            desc=SYSTEMS[name].description, balanced=SYSTEMS[name].balanced, tags=list(SYSTEMS[name].tags),
            labels=bet["labels"], centers=qt.buckets.centers().tolist(), mode=mode,
            two=[_cells(Qp[b], qt.N[b], ROWS_TWO, 1) for b in range(qt.buckets.n)],
            multi=[_cells(Qp[b], qt.N[b], ROWS_MULTI, 0) for b in range(qt.buckets.n)],
            base_two=_best_action(pooled, ROWS_TWO, 1), base_multi=_best_action(pooled, ROWS_MULTI, 0),
            edge=dict(labels=bet["labels"], share=share.tolist(), mean=edge.tolist(), se=se.tolist(),
                      overall=flat["ev_round"], overall_se=float(np.sqrt((share ** 2 * se ** 2).sum()))),
            devs=plays)
        summary.append(dict(name=name, balanced=SYSTEMS[name].balanced, flat=flat["ev_round"],
                            se=systems[name]["edge"]["overall_se"], share=float(share[edge > 0].sum()),
                            gain=flat["ev_round"] - basic,
                            won13=100 * bet["strategies"]["1-3 max EV"]["ev_round"],
                            won03=100 * bet["strategies"]["0-3 max EV"]["ev_round"],
                            ev_hand03=bet["strategies"]["0-3 max EV"]["ev_hand"], seats=bet["seats"]))

    rules = cfg.rules
    solver = InfiniteDeckSolver(rules)
    dp = solver.q_table()
    data = dict(
        title=cfg.title, rounds=cfg.rounds, eval_rounds=cfg.eval_rounds,
        rules={k: getattr(rules, k) for k in rules.__dataclass_fields__},
        ups=["A" if u == 1 else str(u) for u in UPS], rows2=ROWS_TWO, rowsm=ROWS_MULTI,
        dp_two=[[[_r4(x) for x in dp[u - 1, HAND_LABELS.index(r), 1]] for u in UPS] for r in ROWS_TWO],
        dp_multi=[[[_r4(x) for x in dp[u - 1, HAND_LABELS.index(r), 0]] for u in UPS] for r in ROWS_MULTI],
        dp_round_ev=solver.round_ev(), basic_ev=basic, summary=summary, systems=systems, betting=betting,
        seats={s["name"]: s["seats"] for s in summary})

    cfg.docs_dir.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*TITLE*/", f"Counting Strategies, {cfg.title}")
    (cfg.docs_dir / "strategy_card.html").write_text(
        html.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
    _write_csv(summary, basic, cfg.docs_dir / "summary.csv")
    _write_seats_csv(summary, cfg.docs_dir / "by_seat.csv")


def _write_csv(summary, basic, path):
    rows = []
    for s in sorted(summary, key=lambda s: -s["won03"]):
        name, kind = SYSTEMS[s["name"]].description.split(", ", 1)
        rows.append(dict(system=name, type=kind, flat_ev=s["flat"], play_gain=s["gain"],
                         edge_rounds=s["share"], units_per_100_bet_1_3=s["won13"],
                         units_per_100_bet_0_3=s["won03"], ev_per_hand_bet_0_3=s["ev_hand03"]))
    rows.append(dict(system="No count", type="basic strategy", flat_ev=basic, play_gain=0.0))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RECAP_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


SEAT_COLUMNS = ["system", "seat", "flat_ev", "flat_se", "units_per_100_bet_1_3", "units_per_100_bet_0_3"]


def _write_seats_csv(summary, path):
    """Results by seat, 1 = first to act."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SEAT_COLUMNS)
        w.writeheader()
        for s in sorted(summary, key=lambda s: -s["won03"]):
            for i, seat in enumerate(s["seats"]):
                w.writerow(dict(system=SYSTEMS[s["name"]].description.split(", ")[0], seat=i + 1,
                                flat_ev=f"{seat['flat']:.6f}", flat_se=f"{seat['se']:.6f}",
                                units_per_100_bet_1_3=f"{seat['won13']:.6f}",
                                units_per_100_bet_0_3=f"{seat['won03']:.6f}"))


def _seat_table(cfg, system):
    path = cfg.docs_dir / "by_seat.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["system"] == system]
    if len(rows) < 2:
        return []
    lines = ["", f"By seat, {system} (seat 1 acts first; each seat sees the cards of the seats before it):", "",
             "| seat | flat bet EV | bet 1–3, play every round | bet 0–3, sit out bad counts |",
             "|---|---|---|---|"]
    for r in rows:
        cells = [f"{float(r['flat_ev']) * 100:+.3f}% ± {float(r['flat_se']) * 100:.3f}%",
                 f"{float(r['units_per_100_bet_1_3']):+.2f}", f"{float(r['units_per_100_bet_0_3']):+.2f}"]
        lines.append(f"| {r['seat']} | " + " | ".join(c.replace("-", "−") for c in cells) + " |")
    return lines


def _recap_table(cfg):
    def pct(x, d=3):
        return "" if x == "" else f"{float(x) * 100:+.{d}f}%".replace("-", "−")

    def num(x):
        return "" if x == "" else f"{float(x):+.2f}".replace("-", "−")

    lines = [f"### {cfg.title}", "",
             f"[Results page](docs/{cfg.name}/strategy_card.html) · [CSV](docs/{cfg.name}/summary.csv)", "",
             "| system | type | flat bet EV | gain from playing the count | rounds with player edge "
             "| bet 1–3, play every round | bet 0–3, sit out bad counts | 0–3: EV per hand played |",
             "|---|---|---|---|---|---|---|---|"]
    with open(cfg.docs_dir / "summary.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            share = f"{float(r['edge_rounds']) * 100:.1f}%" if r["edge_rounds"] else ""
            lines.append(f"| {r['system']} | {r['type']} | {pct(r['flat_ev'])} | {pct(r['play_gain'])} | {share} "
                         f"| {num(r['units_per_100_bet_1_3'])} | {num(r['units_per_100_bet_0_3'])} "
                         f"| {pct(r['ev_per_hand_bet_0_3'], 2)} |")
    with open(cfg.docs_dir / "summary.csv", encoding="utf-8") as f:
        best = next(csv.DictReader(f))["system"]
    return lines + _seat_table(cfg, best)


def _games_table(configs):
    """One line per game: its best count system (by the 0-3 return) against no count."""
    lines = ["| game | best count | no count: EV per round | best count: flat bet EV "
             "| bet 1–3, play every round | bet 0–3, sit out bad counts | 0–3: EV per hand played |",
             "|---|---|---|---|---|---|---|"]
    for cfg in configs:
        with open(cfg.docs_dir / "summary.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        best, basic = rows[0], rows[-1]
        cells = [f"{float(basic['flat_ev']) * 100:+.2f}%", f"{float(best['flat_ev']) * 100:+.2f}%",
                 f"{float(best['units_per_100_bet_1_3']):+.2f}", f"{float(best['units_per_100_bet_0_3']):+.2f}",
                 f"{float(best['ev_per_hand_bet_0_3']) * 100:+.2f}%"]
        lines.append(f"| [{cfg.title}](docs/{cfg.name}/strategy_card.html) | {best['system']} | "
                     + " | ".join(c.replace("-", "−") for c in cells) + " |")
    return lines


def update_readme(configs, path="README.md"):
    """Rewrite the README recap (between the recap markers) for every config with a summary.csv."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if RECAP_START not in text:
        return False
    configs = [c for c in configs if (c.docs_dir / "summary.csv").exists()]
    lines = [RECAP_START] + _games_table(configs) + [""]
    for cfg in configs:
        lines += _recap_table(cfg) + [""]
    lines += ["Betting columns are units won per 100 rounds dealt, betting 3 units at counts with a "
              "player edge. Generated by `scripts/run.py`.", RECAP_END]
    start, end = text.index(RECAP_START), text.index(RECAP_END) + len(RECAP_END)
    path.write_text(text[:start] + "\n".join(lines) + text[end:], encoding="utf-8")
    return True
