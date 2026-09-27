"""Train, evaluate and report every count system for one or more game configs.

    python scripts/run.py configs/6deck_s17_das.yaml
    python scripts/run.py configs/*.yaml
    python scripts/run.py configs/1deck_h17.yaml --steps report     # rebuild the page only

Finished steps are skipped (their files already exist in results/<game>/), so an
interrupted run resumes; --force redoes them.
"""
import argparse
import glob
import json
import time

from blackjack import report
from blackjack.betting import evaluate_basic_strategy, evaluate_betting
from blackjack.config import Config
from blackjack.qlearn import QTable

STEPS = ("train", "evaluate", "report")


def run(cfg, steps, force):
    out = cfg.results_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"== {cfg.name}: {cfg.title}\n{cfg.rules}", flush=True)
    for name in cfg.systems:
        q_path, bet_path = out / f"q_{name}.npz", out / f"betting_{name}.json"
        if "train" in steps and (force or not q_path.exists()):
            t = time.time()
            QTable.train(cfg.rules, cfg.rounds, count=name, buckets=cfg.buckets(name), verbose=False).save(q_path)
            print(f"{name}: trained in {time.time() - t:.0f}s", flush=True)
        if "evaluate" in steps and (force or not bet_path.exists()):
            t = time.time()
            bet_path.write_text(json.dumps(evaluate_betting(QTable.load(q_path), cfg.eval_rounds), indent=1))
            print(f"{name}: evaluated in {time.time() - t:.0f}s", flush=True)
    if cfg.linear_rounds:
        run_linear(cfg, steps, force)
    basic_path = out / "basic_strategy.json"
    if "evaluate" in steps and (force or not basic_path.exists()):
        qt = QTable.load(out / f"q_{cfg.systems[0]}.npz")
        basic_path.write_text(json.dumps(evaluate_basic_strategy(qt, cfg.eval_rounds)))
    if "report" in steps:
        report.build(cfg)
        print(f"wrote {cfg.docs_dir}/", flush=True)


def run_linear(cfg, steps, force):
    """The agent without a predefined count: a linear model on the full shoe composition."""
    from blackjack.dqn import ModelAgent, train
    out = cfg.results_dir
    path, bet_path = out / "linear.npz", out / "betting_linear.json"
    if "train" in steps and (force or not path.exists()):
        t = time.time()
        behavior = QTable.load(out / f"q_{'hi_lo' if 'hi_lo' in cfg.systems else cfg.systems[0]}.npz")
        net = train(cfg.rules, behavior, cfg.linear_rounds, log=lambda m: None)
        ModelAgent.from_net(net, cfg.rules).save(path)
        print(f"linear: trained in {time.time() - t:.0f}s", flush=True)
    if "evaluate" in steps and (force or not bet_path.exists()):
        t = time.time()
        bet_path.write_text(json.dumps(evaluate_betting(ModelAgent.load(path, cfg.rules), cfg.eval_rounds), indent=1))
        print(f"linear: evaluated in {time.time() - t:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+")
    p.add_argument("--steps", nargs="+", default=list(STEPS), choices=STEPS)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    paths = sorted({f for pattern in a.configs for f in glob.glob(pattern)})
    for path in paths:
        run(Config.load(path), a.steps, a.force)
    if "report" in a.steps:
        all_configs = [Config.load(f) for f in sorted(glob.glob("configs/*.yaml"))]
        if report.update_readme(all_configs):
            print("updated README recap")


if __name__ == "__main__":
    main()
