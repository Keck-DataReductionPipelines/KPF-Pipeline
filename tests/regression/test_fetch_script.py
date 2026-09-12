"""Tests for the fetch scripts layer.

``scripts/fetch/_fetch.py`` holds everything the subject scripts share -- the
flags, the remote listing, the rsync argv and the driver loop -- so it is tested
once here against a stand-in subject. The subject modules themselves only declare
a name and a remote directory, so they are checked as a parametrized set: the
right constants, and that they hand both to the shared driver.

Nothing here reaches the network -- ``subprocess.run`` is stubbed, so the tests
assert on the commands the scripts *would* run. The shared ``--dates`` validation
is tested in test_script_helpers.py.
"""

import pytest

from scripts.fetch import L0, L2, L4, _fetch, masters

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli

# Every subject script, and the two facts each one owns.
SUBJECTS = [
    (masters, "masters", "/data/kpf/vNext/masters"),
    # L0 is the shared raw archive, not a vNext output tree.
    (L0, "L0", "/data/kpf/L0"),
    (L2, "L2", "/data/kpf/vNext/L2"),
    (L4, "L4", "/data/kpf/vNext/L4"),
]

_BASE = ["-u", "someone", "--local_dir", "/out"]


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _parse(argv, subject="L2", remote_dir="/data/kpf/vNext/L2"):
    return _fetch.parse_args(argv, subject, remote_dir, "desc")


class TestParseArgs:
    def test_explicit_dates_are_sorted_and_deduped(self):
        args = _parse([*_BASE, "--dates", "20240712", "20240405", "20240405"])
        assert args.dates == ["20240405", "20240712"]

    def test_dates_file_is_expanded(self, tmp_path):
        listing = tmp_path / "nights.txt"
        listing.write_text("20240405\n\n20240712\n")
        args = _parse([*_BASE, "--dates", str(listing)])
        assert args.dates == ["20240405", "20240712"]

    def test_user_is_required(self):
        with pytest.raises(SystemExit):
            _parse(["--local_dir", "/out", "--dates", "20240405"])

    def test_local_dir_is_required(self):
        with pytest.raises(SystemExit):
            _parse(["-u", "someone", "--dates", "20240405"])

    def test_neither_input_form_is_an_error(self):
        with pytest.raises(SystemExit):
            _parse(_BASE)

    def test_remote_dir_defaults_and_overrides(self):
        assert _parse([*_BASE, "--dates", "20240405"]).remote_dir == (
            "/data/kpf/vNext/L2"
        )
        args = _parse([*_BASE, "--dates", "20240405", "--remote_dir", "/data/x"])
        assert args.remote_dir == "/data/x"

    def test_subject_names_the_program_and_the_remote_dir_help(self):
        with pytest.raises(SystemExit):
            _parse(["--help"], subject="L4", remote_dir="/data/kpf/vNext/L4")


class TestRemoteDatecodes:
    def test_keeps_only_datecodes_in_range(self, monkeypatch):
        listing = "20231231\n20240101\n20240115\nscratch\n20240201\n"
        monkeypatch.setattr(
            _fetch.subprocess, "run", lambda *a, **k: _FakeCompleted(stdout=listing)
        )
        nights = _fetch.remote_datecodes([], "h", "/root", "20240101", "20240131")
        assert nights == ["20240101", "20240115"]

    def test_failed_listing_exits(self, monkeypatch):
        monkeypatch.setattr(
            _fetch.subprocess,
            "run",
            lambda *a, **k: _FakeCompleted(returncode=2, stderr="No such file"),
        )
        with pytest.raises(SystemExit, match="cannot list"):
            _fetch.remote_datecodes([], "h", "/root", "20240101", "20240131")

    def test_empty_range_exits(self, monkeypatch):
        monkeypatch.setattr(
            _fetch.subprocess,
            "run",
            lambda *a, **k: _FakeCompleted(stdout="20250101\n"),
        )
        with pytest.raises(SystemExit, match="no datecode dirs"):
            _fetch.remote_datecodes([], "h", "/root", "20240101", "20240131")


class TestFetchNight:
    def test_builds_rsync_argv(self, monkeypatch):
        seen = []

        def _run(argv, **kwargs):
            seen.append(argv)
            return _FakeCompleted()

        monkeypatch.setattr(_fetch.subprocess, "run", _run)
        assert _fetch.fetch_night(
            ["ssh", "-o", "X=1"], "me@h", "/root", "20240405", "/out"
        )

        argv = seen[0]
        assert argv[0] == "rsync"
        # Trailing slash on the source: the night's *contents* land in the dest dir.
        assert argv[-2] == "me@h:/root/20240405/"
        assert argv[-1] == "/out/20240405"
        # The multiplexed ssh is passed through as one -e string.
        assert argv[argv.index("-e") + 1] == "ssh -o X=1"

    def test_nonzero_rsync_is_a_failure(self, monkeypatch):
        monkeypatch.setattr(
            _fetch.subprocess, "run", lambda *a, **k: _FakeCompleted(returncode=23)
        )
        assert not _fetch.fetch_night([], "me@h", "/root", "20240405", "/out")


class TestMain:
    def _stub(self, monkeypatch, failures=()):
        """Stub every subprocess call; record (remote, night) rsync was asked for."""
        fetched = []

        def _fetch_night(ssh, remote, remote_dir, datecode, local_dir):
            fetched.append((remote, remote_dir, datecode))
            return datecode not in failures

        monkeypatch.setattr(_fetch, "fetch_night", _fetch_night)
        monkeypatch.setattr(_fetch, "close_ssh_connection", lambda *a: None)
        return fetched

    def _run(self, tmp_path, *extra):
        return _fetch.main(
            ["-u", "someone", "--local_dir", str(tmp_path), *extra],
            "L2",
            "/data/kpf/vNext/L2",
            "desc",
        )

    def test_exits_zero_when_all_transferred(self, monkeypatch, tmp_path):
        fetched = self._stub(monkeypatch)
        rc = self._run(tmp_path, "--dates", "20240405", "20240712")
        assert rc == 0
        assert [n for _, _, n in fetched] == ["20240405", "20240712"]

    def test_exits_nonzero_when_any_failed_but_tries_all(self, monkeypatch, tmp_path):
        fetched = self._stub(monkeypatch, failures={"20240405"})
        rc = self._run(tmp_path, "--dates", "20240405", "20240712")
        assert rc == 1
        # fail-soft: both attempted
        assert [n for _, _, n in fetched] == ["20240405", "20240712"]

    def test_user_fills_the_remote_host(self, monkeypatch, tmp_path):
        fetched = self._stub(monkeypatch)
        self._run(tmp_path, "--dates", "20240405")
        assert fetched[0][0] == "someone@shrek.caltech.edu"

    def test_range_resolves_against_the_remote(self, monkeypatch, tmp_path):
        fetched = self._stub(monkeypatch)
        monkeypatch.setattr(
            _fetch, "remote_datecodes", lambda *a: ["20240101", "20240115"]
        )
        rc = self._run(tmp_path, "--date_range", "20240101", "20240131")
        assert rc == 0
        assert [n for _, _, n in fetched] == ["20240101", "20240115"]

    def test_creates_the_local_root(self, monkeypatch, tmp_path):
        self._stub(monkeypatch)
        local = tmp_path / "new" / "tree"
        _fetch.main(
            ["-u", "someone", "--local_dir", str(local), "--dates", "20240405"],
            "L2",
            "/data/kpf/vNext/L2",
            "desc",
        )
        assert local.is_dir()


class TestSubjects:
    """Each subject script owns only its name and its remote tree."""

    @pytest.mark.parametrize("module,subject,remote_dir", SUBJECTS)
    def test_declares_its_subject_and_remote_dir(self, module, subject, remote_dir):
        assert module.SUBJECT == subject
        assert module.DEFAULT_REMOTE_DIR == remote_dir

    @pytest.mark.parametrize("module,subject,remote_dir", SUBJECTS)
    def test_main_hands_both_to_the_shared_driver(
        self, module, subject, remote_dir, monkeypatch
    ):
        seen = {}

        def _main(argv, subj, default_remote_dir, description):
            seen.update(
                argv=argv, subject=subj, remote_dir=default_remote_dir, desc=description
            )
            return 0

        monkeypatch.setattr(_fetch, "main", _main)
        assert module.main(["--dates", "20240405"]) == 0
        assert seen["argv"] == ["--dates", "20240405"]
        assert seen["subject"] == subject
        assert seen["remote_dir"] == remote_dir
        assert subject in seen["desc"]
