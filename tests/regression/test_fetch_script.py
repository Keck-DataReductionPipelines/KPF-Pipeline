"""Tests for the fetch scripts layer.

``scripts/fetch/_fetch.py`` holds everything the subject scripts share -- the
flags, the remote listing, the rsync argv and the driver loop -- so it is tested
once here against a stand-in subject. L0/L2/L4 only declare a name and a remote
directory, so they are checked as a parametrized set: the right constants, and
that they hand both to the shared driver. ``masters`` adds calibration-kind
selection on top, which has its own class.

Nothing here reaches the network -- ``subprocess.run`` is stubbed, so the tests
assert on the commands the scripts *would* run. The shared ``--dates`` validation
is tested in test_script_helpers.py.
"""

import pytest

from scripts._argparse import resolve_dates
from scripts.fetch import L0, L2, L4, _fetch, masters

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli

# The subjects that are nothing but a name and a remote tree. `masters` is not one
# of them -- it adds calibration-kind selection -- so it has its own class below.
SUBJECTS = [
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


def _parse(argv, subject="L2", remote_dir="/data/kpf/vNext/L2", gb_per_night=None):
    ap = _fetch.subject_parser(subject, remote_dir, "desc", gb_per_night)
    return resolve_dates(ap, ap.parse_args(argv))


def _boom(*a, **k):
    raise AssertionError("should not have prompted")


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

        def _fetch_night(ssh, remote, remote_dir, datecode, local_dir, filters=()):
            fetched.append((remote, remote_dir, datecode))
            return datecode not in failures

        monkeypatch.setattr(_fetch, "fetch_night", _fetch_night)
        monkeypatch.setattr(_fetch, "close_ssh_connection", lambda *a: None)
        return fetched

    def _run(self, tmp_path, *extra, gb_per_night=None):
        args = _parse(
            ["-u", "someone", "--local_dir", str(tmp_path), *extra],
            gb_per_night=gb_per_night,
        )
        return _fetch.run(args)

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
        self._run(local, "--dates", "20240405")
        assert local.is_dir()


class TestConfirmVolume:
    """The size gate in front of a large subject (L0). Only subjects that pass a
    per-night estimate get it, so the others are unaffected."""

    def _run(self, monkeypatch, tmp_path, *extra, fetched=None):
        monkeypatch.setattr(
            _fetch, "fetch_night", lambda *a: fetched.append(a[3]) is None or True
        )
        monkeypatch.setattr(_fetch, "close_ssh_connection", lambda *a: None)
        args = _parse(
            ["-u", "someone", "--local_dir", str(tmp_path), *extra],
            subject="L0",
            remote_dir="/data/kpf/L0",
            gb_per_night=70,
        )
        return _fetch.run(args)

    def test_estimate_scales_with_the_night_count(self, monkeypatch, capsys):
        monkeypatch.setattr(_fetch.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _: "y")
        _fetch.confirm_volume("L0", ["20240405", "20240712", "20240713"], 70, "/out")
        out = capsys.readouterr().out
        assert "3 night(s) x ~70 GB = ~210 GB into /out" in out

    def test_yes_answer_continues(self, monkeypatch, tmp_path):
        monkeypatch.setattr(_fetch.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _: "y")
        fetched = []
        rc = self._run(monkeypatch, tmp_path, "--dates", "20240405", fetched=fetched)
        assert rc == 0 and fetched == ["20240405"]

    def test_anything_else_aborts_before_transferring(self, monkeypatch, tmp_path):
        monkeypatch.setattr(_fetch.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _: "")
        fetched = []
        with pytest.raises(SystemExit, match="aborted"):
            self._run(monkeypatch, tmp_path, "--dates", "20240405", fetched=fetched)
        assert fetched == []
        # Aborting must not leave an empty local root behind.
        assert not (tmp_path / "20240405").exists()

    def test_yes_flag_skips_the_prompt(self, monkeypatch, tmp_path):
        monkeypatch.setattr(_fetch.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", _boom)
        fetched = []
        rc = self._run(
            monkeypatch, tmp_path, "--yes", "--dates", "20240405", fetched=fetched
        )
        assert rc == 0 and fetched == ["20240405"]

    def test_non_interactive_run_refuses_without_yes(self, monkeypatch, tmp_path):
        monkeypatch.setattr(_fetch.sys.stdin, "isatty", lambda: False)
        monkeypatch.setattr("builtins.input", _boom)
        with pytest.raises(SystemExit, match="refusing to start unattended"):
            self._run(monkeypatch, tmp_path, "--dates", "20240405", fetched=[])

    def test_subjects_without_an_estimate_have_no_yes_flag(self):
        with pytest.raises(SystemExit):
            _parse([*_BASE, "--dates", "20240405", "--yes"])


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
        # Both facts ride on the namespace, so `run` needs no arguments of its own.
        seen = {}
        monkeypatch.setattr(_fetch, "run", lambda args, **kw: seen.update(vars(args)))
        module.main(["-u", "someone", "--local_dir", "/out", "--dates", "20240405"])
        assert seen["subject"] == subject
        assert seen["remote_dir"] == remote_dir


class TestMastersSelection:
    """`fetch masters` must be told which calibration kinds to pull: a night holds
    every kind together and they differ in size by orders of magnitude."""

    def _parse(self, *extra):
        return masters.parse_args(
            ["-u", "someone", "--local_dir", "/out", "--dates", "20240405", *extra]
        )

    def test_naming_no_kind_is_an_error(self):
        with pytest.raises(SystemExit):
            self._parse()

    def test_all_takes_the_whole_night_unfiltered(self):
        assert masters.rsync_filters(self._parse("--all")) == []

    @pytest.mark.parametrize(
        "flag,expected",
        [
            ("--bias", ["--include=*_master_bias_L1.fits"]),
            ("--dark", ["--include=*_master_dark_L1.fits"]),
            ("--flat", ["--include=*_master_flat_L1.fits"]),
            ("--order_trace", ["--include=*_master_order_trace.csv"]),
            (
                "--thar",
                [
                    "--include=*_master_thar_*",
                    "--include=thar_L2/",
                    "--include=thar_L2/**",
                ],
            ),
        ],
    )
    def test_each_kind_selects_its_own_patterns(self, flag, expected):
        # Everything not named is excluded, so the include list must be exhaustive.
        assert masters.rsync_filters(self._parse(flag)) == expected + ["--exclude=*"]

    def test_kinds_combine(self):
        filters = masters.rsync_filters(self._parse("--flat", "--order_trace"))
        assert filters == [
            "--include=*_master_flat_L1.fits",
            "--include=*_master_order_trace.csv",
            "--exclude=*",
        ]

    def test_thar_brings_the_sidecar_and_diagnostics(self):
        filters = masters.rsync_filters(self._parse("--thar"))
        # The master L2 and the diagnostics .h5 share the _master_thar_ stem.
        assert "--include=*_master_thar_*" in filters
        # The sidecar dir needs both rules to be descended into.
        assert "--include=thar_L2/" in filters
        assert "--include=thar_L2/**" in filters

    @pytest.mark.parametrize("kind", ["lfc", "etalon"])
    def test_future_wls_kinds_follow_the_thar_pattern(self, kind):
        filters = masters.rsync_filters(self._parse(f"--{kind}"))
        assert filters == [
            f"--include=*_master_{kind}_*",
            f"--include={kind}_L2/",
            f"--include={kind}_L2/**",
            "--exclude=*",
        ]

    def test_filters_reach_the_driver(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            _fetch, "run", lambda args, **kw: seen.update(kw, subject=args.subject) or 0
        )
        masters.main(
            ["-u", "someone", "--local_dir", "/out", "--dates", "20240405", "--bias"]
        )
        assert seen["subject"] == "masters"
        assert seen["filters"] == [
            "--include=*_master_bias_L1.fits",
            "--exclude=*",
        ]
