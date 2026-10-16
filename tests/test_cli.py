import json
import sys
import types
from pathlib import Path

import pytest

from xmprobe import cli
from xmprobe.common import GateFailed, is_done, mark_done, stamp_key
from xmprobe.config import load_config, with_overrides

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path):
    base = load_config(ROOT / "configs/qwen35_0p8b.yaml")
    return with_overrides(base, **{"paths.scratch_root": str(tmp_path)})


def _fake_stage(monkeypatch, name, fn):
    mod = types.ModuleType(f"fake_{name}")
    mod.run = fn
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setitem(cli.STAGES, name, mod.__name__)


def test_manifest_has_required_fields(cfg, monkeypatch):
    _fake_stage(monkeypatch, "noop", lambda cfg, limit=None, force=False: {"n": 3})
    cli.run_stage("noop", cfg, limit=10)
    m = json.loads(cli.manifest_path(cfg).read_text())
    for field in ("git_commit", "config", "packages", "model_id", "model_revision", "gpu",
                  "slurm_job_id", "created", "updated"):
        assert field in m
    for pkg in ("torch", "transformers", "scikit-learn", "pillow"):
        assert pkg in m["packages"]
    assert m["font_sha256"] and len(m["font_sha256"]) == 64
    rec = m["stages"]["noop"]
    assert rec["status"] == "ok" and rec["n"] == 3 and rec["limit"] == 10
    assert "wall_seconds" in rec


def test_gate_failure_recorded_and_all_stops(cfg, monkeypatch, tmp_path):
    calls = []

    def ok(name):
        return lambda cfg, limit=None, force=False: calls.append(name) or {}

    def gate(cfg, limit=None, force=False):
        calls.append("ocr_check")
        raise GateFailed("ocr", "median_cer 0.3 > 0.10")

    for name in cli.ALL_ORDER:
        _fake_stage(monkeypatch, name, gate if name == "ocr_check" else ok(name))
    cfg_path = tmp_path / "c.yaml"
    import yaml
    cfg_path.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}))
    assert cli.main(["all", "--config", str(cfg_path)]) == 3
    assert calls == ["validate_data", "render", "ocr_check"]
    m = json.loads(cli.manifest_path(cfg).read_text())
    assert m["stages"]["ocr_check"]["status"] == "gate_failed"
    assert m["stages"]["ocr_check"]["gate"] == "ocr"


def test_stamp_skip_and_invalidation(cfg, tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "a.png").write_bytes(b"x")
    key = stamp_key(cfg, ("render",), None)
    assert not is_done(out, "render", key)
    mark_done(out, "render", key, ["a.png"])
    assert is_done(out, "render", key)
    # different --limit or render settings invalidate the stamp
    assert not is_done(out, "render", stamp_key(cfg, ("render",), 10))
    changed = with_overrides(cfg, **{"render.font_size": 28})
    assert not is_done(out, "render", stamp_key(changed, ("render",), None))
    # missing output invalidates the stamp
    (out / "a.png").unlink()
    assert not is_done(out, "render", key)


def test_manifest_revision_follows_current_config(cfg, monkeypatch):
    _fake_stage(monkeypatch, "noop", lambda cfg, limit=None, force=False: {})
    cli.run_stage("noop", with_overrides(cfg, **{"model.revision": "main"}))
    cli.run_stage("noop", with_overrides(cfg, **{"model.revision": "abc123"}))
    assert json.loads(cli.manifest_path(cfg).read_text())["model_revision"] == "abc123"
