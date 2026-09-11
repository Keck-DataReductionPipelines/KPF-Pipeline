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
