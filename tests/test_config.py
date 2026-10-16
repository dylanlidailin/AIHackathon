from pathlib import Path

import pytest
import yaml

from xmprobe.config import ConfigError, config_hash, load_config, with_overrides

ROOT = Path(__file__).resolve().parents[1]


def test_reordered_keys_same_hash(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/qwen35_2b.yaml").read_text())
    reordered = {k: cfg[k] for k in reversed(list(cfg))}
    reordered["render"] = {k: cfg["render"][k] for k in reversed(list(cfg["render"]))}
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text(yaml.safe_dump(cfg, sort_keys=False))
    b.write_text(yaml.safe_dump(reordered, sort_keys=False))
    assert load_config(a)["_hash"] == load_config(b)["_hash"]


def test_value_change_changes_hash():
    cfg = load_config(ROOT / "configs/qwen35_2b.yaml")
    assert with_overrides(cfg, **{"render.font_size": 28})["_hash"] != cfg["_hash"]


def test_section_hash_ignores_other_sections():
    cfg = load_config(ROOT / "configs/qwen35_2b.yaml")
    other = with_overrides(cfg, **{"probe.bootstrap": 10})
    assert config_hash(cfg, ("render",)) == config_hash(other, ("render",))


@pytest.mark.parametrize("name", ["qwen35_2b", "qwen35_0p8b", "qwen25vl_3b"])
def test_shipped_configs_load(name):
    cfg = load_config(ROOT / f"configs/{name}.yaml")
    assert cfg["run_name"].startswith(name)
    assert cfg["probe"]["methods"] == ["lr", "dom"]


def test_missing_section_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("run_name: x\n")
    with pytest.raises(ConfigError, match="missing sections"):
        load_config(p)
