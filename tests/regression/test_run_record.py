"""Tests for kpfpipe/utils/run_record.py: the per-invocation run.json sidecar.

The sidecar sits beside the log file (same directory, same stem) so a log and its
run record are always found together; writes are atomic so a reader never sees a
half-written file; git_sha never raises (a damaged or absent .git records null).
"""

import json
import os
import subprocess

import pytest

from kpfpipe.utils import run_record as rr


class TestRunJsonPath:
    def test_replaces_log_suffix(self):
        assert (
            rr.run_json_path("/l/20240405/kpf_science_KP.1_20240405T010203.log")
            == "/l/20240405/kpf_science_KP.1_20240405T010203.run.json"
        )

    def test_keeps_collision_suffix(self):
        assert (
            rr.run_json_path("/l/d/kpf_masters_batch_x.log.3")
            == "/l/d/kpf_masters_batch_x.3.run.json"
        )

    def test_rejects_non_log_path(self):
        with pytest.raises(ValueError):
            rr.run_json_path("/l/d/notalog.txt")


class TestGitSha:
    def test_returns_head_sha_of_a_repo(self, tmp_path):
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "commit",
                "-q",
                "--allow-empty",
                "-m",
                "x",
            ],
            check=True,
        )
        sha = rr.git_sha(tmp_path)
        assert isinstance(sha, str) and len(sha) == 40

    def test_returns_none_outside_a_repo(self, tmp_path):
        assert rr.git_sha(tmp_path) is None

    def test_returns_none_when_git_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rr.shutil, "which", lambda name: None)
        assert rr.git_sha(tmp_path) is None


class TestAtomicJson:
    def test_write_then_read_roundtrip(self, tmp_path):
        path = str(tmp_path / "a.run.json")
        rr.write_json_atomic(path, {"schema": 1, "status": "running"})
        assert rr.read_run_record(path) == {"schema": 1, "status": "running"}

    def test_creates_parent_dir(self, tmp_path):
        path = str(tmp_path / "sub" / "a.run.json")
        rr.write_json_atomic(path, {"x": 1})
        assert os.path.isfile(path)

    def test_no_tmp_file_left_behind(self, tmp_path):
        path = str(tmp_path / "a.run.json")
        rr.write_json_atomic(path, {"x": 1})
        assert os.listdir(tmp_path) == ["a.run.json"]

    def test_read_rejects_wrong_schema(self, tmp_path):
        path = tmp_path / "a.run.json"
        path.write_text(json.dumps({"schema": 99}))
        with pytest.raises(ValueError):
            rr.read_run_record(str(path))


def _log(tmp_path):
    d = tmp_path / "20240405"
    d.mkdir(exist_ok=True)
    return str(d / "kpf_science_KP.20240405.1_20240405T010203.log")


class TestRunRecordStart:
    def test_writes_running_record_beside_log(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rr, "git_sha", lambda repo_root=None: "abc123")
        monkeypatch.delenv(rr.PARENT_ENV, raising=False)
        monkeypatch.delenv(rr.FLOW_RUN_ENV, raising=False)
        rec = rr.RunRecord.start(
            _log(tmp_path),
            kind="run",
            recipe="science",
            target="KP.20240405.1",
            config="/c/s.toml",
            argv=["kpfpipe", "run"],
        )
        assert rec.path == rr.run_json_path(_log(tmp_path))
        data = rr.read_run_record(rec.path)
        assert data["schema"] == rr.SCHEMA
        assert data["kind"] == "run"
        assert data["status"] == "running"
        assert data["recipe"] == "science"
        assert data["target"] == "KP.20240405.1"
        assert data["config"] == "/c/s.toml"
        assert data["argv"] == ["kpfpipe", "run"]
        assert data["command"] == "kpfpipe run"
        assert data["git_sha"] == "abc123"
        assert data["drptag"] == rr.kpfpipe.__version__
        assert data["host"] == rr.socket.gethostname()
        assert data["pid"] == os.getpid()
        assert data["log_path"] == _log(tmp_path)
        assert data["parent"] is None
        assert data["flow_run_id"] is None
        assert data["ended_utc"] is None
        assert data["exit_status"] is None
        assert data["counts"] == {"done": 0, "failed": 0, "skipped": 0}
        assert data["children"] == []
        assert data["started_utc"].endswith("Z")
        rec.finish(0)  # detach atexit

    def test_records_parent_and_flow_run_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rr, "git_sha", lambda repo_root=None: None)
        monkeypatch.setenv(rr.PARENT_ENV, "/l/d/kpf_science_batch_x.run.json")
        monkeypatch.setenv(rr.FLOW_RUN_ENV, "flow-uuid")
        rec = rr.RunRecord.start(
            _log(tmp_path), kind="run", recipe="science", target="t", config="c"
        )
        data = rr.read_run_record(rec.path)
        assert data["parent"] == "/l/d/kpf_science_batch_x.run.json"
        assert data["flow_run_id"] == "flow-uuid"
        assert data["git_sha"] is None
        rec.finish(0)

    def test_argv_defaults_to_sys_argv(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rr, "git_sha", lambda repo_root=None: None)
        monkeypatch.setattr(rr.sys, "argv", ["prog", "-o", "x"])
        rec = rr.RunRecord.start(
            _log(tmp_path), kind="run", recipe="science", target="t", config="c"
        )
        assert rr.read_run_record(rec.path)["argv"] == ["prog", "-o", "x"]
        rec.finish(0)

    def test_bad_kind_raises(self, tmp_path):
        with pytest.raises(ValueError):
            rr.RunRecord.start(
                _log(tmp_path), kind="nope", recipe="science", target="t", config="c"
            )


class TestRunRecordFinish:
    def _start(self, tmp_path, monkeypatch, kind="run"):
        monkeypatch.setattr(rr, "git_sha", lambda repo_root=None: None)
        return rr.RunRecord.start(
            _log(tmp_path), kind=kind, recipe="science", target="t", config="c"
        )

    def test_exit_zero_is_succeeded_with_counts(self, tmp_path, monkeypatch):
        rec = self._start(tmp_path, monkeypatch)
        rec.finish(0, done=1)
        data = rr.read_run_record(rec.path)
        assert data["status"] == "succeeded"
        assert data["exit_status"] == 0
        assert data["counts"] == {"done": 1, "failed": 0, "skipped": 0}
        assert data["ended_utc"] is not None
        assert rec.is_finished()

    def test_nonzero_is_failed(self, tmp_path, monkeypatch):
        rec = self._start(tmp_path, monkeypatch)
        rec.finish(1, failed=1)
        assert rr.read_run_record(rec.path)["status"] == "failed"

    def test_children_recorded_for_batch(self, tmp_path, monkeypatch):
        rec = self._start(tmp_path, monkeypatch, kind="batch")
        kids = [{"tag": "KP.1", "exit_status": 0, "run_json": "/l/d/a.run.json"}]
        rec.finish(0, done=1, children=kids)
        assert rr.read_run_record(rec.path)["children"] == kids

    def test_finish_twice_raises(self, tmp_path, monkeypatch):
        rec = self._start(tmp_path, monkeypatch)
        rec.finish(0)
        with pytest.raises(RuntimeError):
            rec.finish(0)

    def test_atexit_marks_unfinished_record_interrupted(self, tmp_path, monkeypatch):
        registered = []
        monkeypatch.setattr(rr.atexit, "register", lambda fn: registered.append(fn))
        rec = self._start(tmp_path, monkeypatch)
        assert len(registered) == 1
        registered[0]()  # simulate interpreter exit without finish()
        data = rr.read_run_record(rec.path)
        assert data["status"] == "interrupted"
        assert data["exit_status"] is None
        assert data["ended_utc"] is not None

    def test_atexit_is_noop_after_finish(self, tmp_path, monkeypatch):
        registered = []
        monkeypatch.setattr(rr.atexit, "register", lambda fn: registered.append(fn))
        rec = self._start(tmp_path, monkeypatch)
        rec.finish(0)
        before = rr.read_run_record(rec.path)
        registered[0]()
        assert rr.read_run_record(rec.path) == before


class TestCollectChildRecords:
    def _child(self, log_dir, night, tag, parent, status="succeeded", rc=0):
        p = os.path.join(log_dir, night, f"kpf_science_{tag}_20240405T000000.run.json")
        rr.write_json_atomic(
            p,
            {
                "schema": rr.SCHEMA,
                "kind": "run",
                "status": status,
                "target": tag,
                "exit_status": rc,
                "parent": parent,
            },
        )
        return p

    def test_returns_children_pointing_at_parent_sorted(self, tmp_path):
        log_dir = str(tmp_path)
        parent = os.path.join(log_dir, "20240405", "kpf_science_batch_x.run.json")
        b = self._child(log_dir, "20240405", "KP.2", parent, "failed", 1)
        a = self._child(log_dir, "20240405", "KP.1", parent)
        self._child(log_dir, "20240405", "KP.9", "/elsewhere/other.run.json")
        self._child(log_dir, "20240406", "KP.3", None)
        assert rr.collect_child_records(log_dir, parent) == [
            {"tag": "KP.1", "exit_status": 0, "status": "succeeded", "run_json": a},
            {"tag": "KP.2", "exit_status": 1, "status": "failed", "run_json": b},
        ]

    def test_skips_unreadable_or_foreign_json(self, tmp_path):
        log_dir = str(tmp_path)
        parent = "/l/p.run.json"
        (tmp_path / "20240405").mkdir()
        (tmp_path / "20240405" / "junk.run.json").write_text("{not json")
        (tmp_path / "20240405" / "other.run.json").write_text('{"schema": 99}')
        good = self._child(log_dir, "20240405", "KP.1", parent)
        got = rr.collect_child_records(log_dir, parent)
        assert [c["run_json"] for c in got] == [good]

    def test_missing_log_dir_returns_empty(self, tmp_path):
        assert rr.collect_child_records(str(tmp_path / "nope"), "/l/p.run.json") == []
