"""Tests for scripts/fetch/masters.py: the masters fetch wrapper.

Covers the script's own surface: the night selection it shares with
``kpfpipe masters``, the remote listing it parses into datecodes, and the rsync/ssh
argv it builds. Nothing here reaches the network -- ``subprocess.run`` is stubbed,
so the tests assert on the commands the script *would* run. The shared ``--dates``
validation is tested in test_script_helpers.py.
"""

import pytest

from scripts.fetch import masters as _fetch_masters

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli


@pytest.fixture(scope="module")
def f():
    return _fetch_masters


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


_BASE = ["-u", "someone", "--local_dir", "/out"]


class TestParseArgs:
    def test_explicit_dates_are_sorted_and_deduped(self, f):
        args = f.parse_args([*_BASE, "--dates", "20240712", "20240405", "20240405"])
        assert args.dates == ["20240405", "20240712"]

    def test_dates_file_is_expanded(self, f, tmp_path):
        listing = tmp_path / "nights.txt"
        listing.write_text("20240405\n\n20240712\n")
        args = f.parse_args([*_BASE, "--dates", str(listing)])
        assert args.dates == ["20240405", "20240712"]

    def test_user_is_required(self, f):
        with pytest.raises(SystemExit):
            f.parse_args(["--local_dir", "/out", "--dates", "20240405"])

    def test_local_dir_is_required(self, f):
        with pytest.raises(SystemExit):
            f.parse_args(["-u", "someone", "--dates", "20240405"])

    def test_neither_input_form_is_an_error(self, f):
        with pytest.raises(SystemExit):
            f.parse_args(_BASE)

    def test_remote_dir_defaults_but_is_overridable(self, f):
        args = f.parse_args([*_BASE, "--dates", "20240405"])
        assert args.remote_dir == f.DEFAULT_REMOTE_DIR

        args = f.parse_args([*_BASE, "--dates", "20240405", "--remote_dir", "/data/m"])
        assert args.remote_dir == "/data/m"

    def test_user_fills_the_remote_host(self, f, monkeypatch, tmp_path):
        seen = []
        monkeypatch.setattr(
            f, "fetch_night", lambda ssh, remote, *a: seen.append(remote) or True
        )
        monkeypatch.setattr(f, "close_ssh_connection", lambda *a: None)
        f.main(["-u", "someone", "--local_dir", str(tmp_path), "--dates", "20240405"])
        assert seen == ["someone@shrek.caltech.edu"]


class TestRemoteDatecodes:
    def test_keeps_only_datecodes_in_range(self, f, monkeypatch):
        listing = "20231231\n20240101\n20240115\nscratch\n20240201\n"
        monkeypatch.setattr(
            f.subprocess, "run", lambda *a, **k: _FakeCompleted(stdout=listing)
        )
        nights = f.remote_datecodes([], "h", "/root", "20240101", "20240131")
        assert nights == ["20240101", "20240115"]

    def test_failed_listing_exits(self, f, monkeypatch):
        monkeypatch.setattr(
            f.subprocess,
            "run",
            lambda *a, **k: _FakeCompleted(returncode=2, stderr="No such file"),
        )
        with pytest.raises(SystemExit, match="cannot list"):
            f.remote_datecodes([], "h", "/root", "20240101", "20240131")

    def test_empty_range_exits(self, f, monkeypatch):
        monkeypatch.setattr(
            f.subprocess, "run", lambda *a, **k: _FakeCompleted(stdout="20250101\n")
        )
        with pytest.raises(SystemExit, match="no datecode dirs"):
            f.remote_datecodes([], "h", "/root", "20240101", "20240131")


class TestFetchNight:
    def test_builds_rsync_argv(self, f, monkeypatch):
        seen = []

        def _run(argv, **kwargs):
            seen.append(argv)
            return _FakeCompleted()

        monkeypatch.setattr(f.subprocess, "run", _run)
        assert f.fetch_night(["ssh", "-o", "X=1"], "me@h", "/root", "20240405", "/out")

        argv = seen[0]
        assert argv[0] == "rsync"
        # Trailing slash on the source: the night's *contents* land in the dest dir.
        assert argv[-2] == "me@h:/root/20240405/"
        assert argv[-1] == "/out/20240405"
        # The multiplexed ssh is passed through as one -e string.
        assert argv[argv.index("-e") + 1] == "ssh -o X=1"

    def test_nonzero_rsync_is_a_failure(self, f, monkeypatch):
        monkeypatch.setattr(
            f.subprocess, "run", lambda *a, **k: _FakeCompleted(returncode=23)
        )
        assert not f.fetch_night([], "me@h", "/root", "20240405", "/out")


class TestMain:
    def _stub(self, f, monkeypatch, failures=()):
        """Stub every subprocess call; record the nights rsync was asked for."""
        fetched = []

        def _fetch(ssh, remote, remote_dir, datecode, local_dir):
            fetched.append(datecode)
            return datecode not in failures

        monkeypatch.setattr(f, "fetch_night", _fetch)
        monkeypatch.setattr(f, "close_ssh_connection", lambda *a: None)
        return fetched

    def test_exits_zero_when_all_transferred(self, f, monkeypatch, tmp_path):
        fetched = self._stub(f, monkeypatch)
        rc = f.main(
            ["-u", "someone", "--local_dir", str(tmp_path),
             "--dates", "20240405", "20240712"]
        )  # fmt: skip
        assert rc == 0
        assert fetched == ["20240405", "20240712"]

    def test_exits_nonzero_when_any_failed_but_tries_all(
        self, f, monkeypatch, tmp_path
    ):
        fetched = self._stub(f, monkeypatch, failures={"20240405"})
        rc = f.main(
            ["-u", "someone", "--local_dir", str(tmp_path),
             "--dates", "20240405", "20240712"]
        )  # fmt: skip
        assert rc == 1
        assert fetched == ["20240405", "20240712"]  # fail-soft: both attempted

    def test_range_resolves_against_the_remote(self, f, monkeypatch, tmp_path):
        fetched = self._stub(f, monkeypatch)
        monkeypatch.setattr(f, "remote_datecodes", lambda *a: ["20240101", "20240115"])
        rc = f.main(
            ["-u", "someone", "--local_dir", str(tmp_path),
             "--date_range", "20240101", "20240131"]
        )  # fmt: skip
        assert rc == 0
        assert fetched == ["20240101", "20240115"]

    def test_creates_the_local_root(self, f, monkeypatch, tmp_path):
        self._stub(f, monkeypatch)
        local = tmp_path / "new" / "masters"
        f.main(["-u", "someone", "--local_dir", str(local), "--dates", "20240405"])
        assert local.is_dir()
