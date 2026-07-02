"""The iteration tooling: dev.py task table, eval report artifact, the
run_episode close-ordering, and DAgger collection (torch-only)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load_dev():
    spec = importlib.util.spec_from_file_location("dev", REPO / "scripts" / "dev.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dev_task_table_is_wellformed():
    dev = _load_dev()
    expected = {"preflight", "test", "gif", "usd", "walk-smoke", "walk",
                "watch", "isaac", "demos", "dagger", "vla", "eval"}
    assert expected <= set(dev.TASKS)
    for name, (cmd, desc) in dev.TASKS.items():
        assert cmd[0] == sys.executable and all(isinstance(c, str) for c in cmd), name
        assert desc
    assert dev.main(["list"]) == 0
    assert dev.main(["not-a-task"]) == 2


def test_eval_report_writer(tmp_path):
    from gardener_bdx.training.evaluate import write_report

    a = {"episodes": 2, "success_rate": 0.5, "serviced": 1.0, "thirsty": 2.0,
         "water_l": 0.3, "collisions": 0.0, "safety_s": 1.2,
         "complete_rate": 0.5, "time_to_complete_s": 12.0}
    jp, mp = write_report(str(tmp_path / "report"), {"teacher": a, "learned": a},
                          meta={"backend": "kinematic"})
    payload = json.loads(Path(jp).read_text())
    assert payload["results"]["teacher"]["success_rate"] == 0.5
    assert "git" in payload["meta"] and "backend" in payload["meta"]
    md = Path(mp).read_text()
    assert "| Success rate | 50.0% | 50.0% |" in md


def test_run_episode_final_state_read_before_close():
    """Regression: plant dryness must be captured before io.close() (the Isaac
    backend tears down the app on close)."""
    from gardener_bdx.common.config import HierarchyConfig, RobotConfig
    from gardener_bdx.training.evaluate import run_episode

    rc, h = RobotConfig.from_yaml(), HierarchyConfig.from_yaml()
    r = run_episode("kinematic", "scripted", "water the thirsty plants",
                    steps=120, seed=3, vla_kwargs={}, dt=1.0 / h.locomotion_hz, rc=rc, h=h)
    assert 0 <= r.serviced <= r.thirsty
    assert r.steps == 120


def test_dataset_merge_offsets_episodes(tmp_path):
    from gardener_bdx.training.train_vla import _load_datasets

    def fake(n, with_optional):
        d = {"proprio": np.zeros((n, 32), np.float32), "tokens": np.zeros((n, 16), np.int64),
             "actions": np.zeros((n, 7), np.float32), "skills": np.zeros(n, np.int64),
             "images": np.zeros((n, 48, 64, 3), np.uint8),
             "instructions": np.array(["x"] * n), "episode_ends": np.array([n], np.int64)}
        if with_optional:
            d["depths"] = np.ones((n, 48, 64), np.float32)
            d["bev"] = np.ones((n, 64, 64), np.float32)
            d["expressions"] = np.ones(n, np.int64)
        return d

    p1, p2 = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(p1, **fake(10, True))
    np.savez(p2, **fake(6, False))  # e.g. an older BC set without depth/BEV
    d = _load_datasets(f"{p1},{p2}")
    assert d["actions"].shape[0] == 16
    assert d["episode_ends"].tolist() == [10, 16]        # offsets applied
    assert d["depths"].shape == (16, 48, 64)             # zero-filled tail
    assert float(d["depths"][:10].sum()) > 0 and float(d["depths"][10:].sum()) == 0
    assert d["expressions"][:10].sum() == 10 and d["expressions"][10:].sum() == 0
