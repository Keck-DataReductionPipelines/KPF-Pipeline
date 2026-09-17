"""RUN_START / RUN_END provenance lines (kpfpipe.utils.run_events)."""

import json
import logging
import subprocess

import pytest

import kpfpipe
from kpfpipe.utils import run_events as re_


@pytest.fixture
def capture(caplog):
    caplog.set_level(logging.INFO, logger=re_.logger.name)
    return caplog


def _lines(caplog):
    return [r.getMessage() for r in caplog.records]


class TestGitSha:
    def test_returns_hex_for_this_repo(self):
        sha = re_.git_sha()
        assert sha is None or (len(sha) == 40 and int(sha, 16) >= 0)

    def test_none_when_git_fails(self, tmp_path):
        assert re_.git_sha(tmp_path) is None

    def test_none_when_git_missing(self, monkeypatch):
        def boom(*a, **k):
            raise FileNotFoundError("git")

        monkeypatch.setattr(subprocess, "run", boom)
        assert re_.git_sha() is None


class TestRunEvents:
    def test_start_emits_run_start_with_fields(self, capture, monkeypatch):
        monkeypatch.setattr(re_, "git_sha", lambda repo_root=None: "deadbeef")
        ev = re_.RunEvents.start(
            "/l/x.log",
            kind="run",
            recipe="rec",
            target="KP.1",
            config="c.toml",
            parent="/l/batch.log",
            flow_run_id="flow-1",
            argv=["kpfpipe", "run", "-o", "KP.1"],
        )
        (line,) = _lines(capture)
        assert line.startswith(re_.START_TAG)
        start = json.loads(line[len(re_.START_TAG) :])
        assert start["schema"] == re_.SCHEMA
        assert start["kind"] == "run" and start["recipe"] == "rec"
        assert start["target"] == "KP.1" and start["config"] == "c.toml"
        assert start["command"] == "kpfpipe run -o KP.1"
        assert start["git_sha"] == "deadbeef"
        assert start["drptag"] == kpfpipe.__version__
        assert start["log_path"] == "/l/x.log"
        assert start["parent"] == "/l/batch.log"
        assert start["flow_run_id"] == "flow-1"
        assert start["started_utc"].endswith("Z")
        assert not ev.is_finished()

    def test_no_parent_and_no_flow_are_null(self, capture):
        re_.RunEvents.start(
            "/l/x.log", kind="run", recipe="r", target="t", config="c", argv=["x"]
        )
        start, _ = re_.parse_run_events(_lines(capture))
        assert start["parent"] is None and start["flow_run_id"] is None

    def test_finish_success_and_failure(self, capture):
        ev = re_.RunEvents.start(
            "/l/x.log", kind="batch", recipe="science", target="batch", config="c"
        )
        ev.finish(0, done=3, skipped=1)
        _, end = re_.parse_run_events(_lines(capture))
        assert end["status"] == "succeeded" and end["exit_status"] == 0
        assert end["counts"] == {"done": 3, "failed": 0, "skipped": 1}
        assert end["ended_utc"].endswith("Z")
        assert ev.is_finished()

        capture.clear()
        ev2 = re_.RunEvents.start(
            "/l/y.log", kind="run", recipe="r", target="t", config="c"
        )
        ev2.finish(1, failed=1)
        _, end = re_.parse_run_events(_lines(capture))
        assert end["status"] == "failed" and end["exit_status"] == 1

    def test_finish_twice_raises(self, capture):
        ev = re_.RunEvents.start(
            "/l/x.log", kind="run", recipe="r", target="t", config="c"
        )
        ev.finish(0)
        with pytest.raises(RuntimeError):
            ev.finish(0)

    def test_atexit_hook_marks_interrupted_once(self, capture):
        ev = re_.RunEvents.start(
            "/l/x.log", kind="run", recipe="r", target="t", config="c"
        )
        ev._on_exit()
        _, end = re_.parse_run_events(_lines(capture))
        assert end["status"] == "interrupted" and end["exit_status"] is None
        assert end["counts"] is None
        n = len(_lines(capture))
        ev._on_exit()  # idempotent after the first
        assert len(_lines(capture)) == n

    def test_bad_kind_rejected(self):
        with pytest.raises(ValueError, match="kind"):
            re_.RunEvents.start(
                "/l/x.log", kind="wat", recipe="r", target="t", config="c"
            )


class TestParseRunEvents:
    def test_parses_formatted_log_lines(self):
        lines = [
            "2024-09-24T01:00:00.000Z INFO     kpfpipe.utils.run_events: "
            'RUN_START {"kind": "run", "schema": 1}\n',
            "2024-09-24T01:00:01.000Z INFO     kpfpipe.recipe: working\n",
            "2024-09-24T01:05:00.000Z INFO     kpfpipe.utils.run_events: "
            'RUN_END {"status": "succeeded"}\n',
        ]
        start, end = re_.parse_run_events(lines)
        assert start == {"kind": "run", "schema": 1}
        assert end == {"status": "succeeded"}

    def test_missing_end_and_garbage(self):
        start, end = re_.parse_run_events(["x RUN_START {not json", "plain"])
        assert start is None and end is None
        start, end = re_.parse_run_events(['RUN_START {"a": 1}'])
        assert start == {"a": 1} and end is None
