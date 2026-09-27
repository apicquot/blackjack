import glob
from pathlib import Path

import pytest

from blackjack import report
from blackjack.config import Config

ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize("path", sorted(glob.glob(str(ROOT / "configs" / "*.yaml"))))
def test_configs_load(path):
    cfg = Config.load(path)
    assert cfg.systems and cfg.rounds > 0 and cfg.title


def test_bad_config_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("title: x\nrules:\n  decks: 2\n")
    with pytest.raises(ValueError):
        Config.load(bad)


def test_readme_recap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "g.yaml"
    cfg_path.write_text("title: Test game\nrules:\n  n_decks: 2\nsystems: [hi_lo, ko]\n")
    cfg = Config.load(cfg_path)
    cfg.docs_dir.mkdir(parents=True)
    (cfg.docs_dir / "summary.csv").write_text(
        ",".join(report.RECAP_COLUMNS) + "\n"
        "Hi-Lo,level 1,-0.3,0.1,0.36,0.13,0.80,2.2\n"
        "Knock-Out,level 1,-0.4,0.05,0.26,0.08,0.75,2.9\n"
        "No count,basic strategy,-0.5,0.0,,,,\n")
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{report.RECAP_START}\nold\n{report.RECAP_END}\noutro\n")
    assert report.update_readme([cfg], readme)
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text and text.startswith("intro") and text.endswith("outro\n")
    assert "| [Test game](docs/g/strategy_card.html) | Hi-Lo | −0.50 | −0.30 | +0.13 | +0.80 | +2.20 |" in text
    assert "| Knock-Out | level 1 | −0.400 | +0.050 | 26.0% | +0.08 | +0.75 | +2.90 |" in text
    assert "| No count | basic strategy | −0.500 | +0.000 |  |  |  |  |" in text
