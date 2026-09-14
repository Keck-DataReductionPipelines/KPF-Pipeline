"""Tests for tools/cli.py: the ``kpfpipe`` subcommand dispatcher.

tools/cli.py is a thin, git-style router: it maps the first argument to a
subcommand under ``scripts/process/`` and forwards the remaining argv verbatim.
Only the routing, usage banner, and unknown-command error are covered here; the
subcommands' own parsing lives in test_{reduce,masters,science}_script.py.
"""

import pytest

from scripts.fetch import L0 as fetch_l0
from scripts.fetch import L2 as fetch_l2
from scripts.fetch import L4 as fetch_l4
from scripts.fetch import masters as fetch_masters
from tools import cli

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli


class TestDispatch:
    @pytest.mark.parametrize("command", ["run", "masters", "science", "timeseries"])
    def test_routes_to_command_with_forwarded_argv(self, command, monkeypatch):
        seen = []
        monkeypatch.setitem(cli._COMMANDS, command, lambda rest: seen.append(rest))
        rest = ["--science", "-o", "KP.x", "--log_level", "DEBUG"]
        cli.main([command, *rest])
        assert seen == [rest]

    def test_return_value_propagates(self, monkeypatch):
        monkeypatch.setitem(cli._COMMANDS, "run", lambda rest: 7)
        assert cli.main(["run", "-o", "KP.x"]) == 7


class TestUsage:
    def test_no_args_prints_usage_and_returns_zero(self, capsys):
        assert cli.main([]) == 0
        out = capsys.readouterr().out
        assert "usage: kpfpipe <command>" in out
        assert "run" in out and "masters" in out and "science" in out
        assert "timeseries" in out

    @pytest.mark.parametrize("flag", ["-h", "--help"])
    def test_help_flag_prints_usage(self, flag, capsys):
        assert cli.main([flag]) == 0
        assert "usage: kpfpipe <command>" in capsys.readouterr().out


class TestRealtimeCommand:
    def test_realtime_is_routed_and_listed(self, monkeypatch, capsys):
        seen = []
        monkeypatch.setitem(cli._COMMANDS, "realtime", lambda rest: seen.append(rest))
        cli.main(["realtime", "--once"])
        assert seen == [["--once"]]
        cli.main([])
        assert "  realtime    " in capsys.readouterr().out


class TestFetchDispatch:
    """The ``fetch`` router. Nothing here reaches the network: every subject is
    replaced by a stub, or the lookup fails before a script is called."""

    def test_routes_to_subject_with_forwarded_argv(self, monkeypatch):
        seen = []
        monkeypatch.setitem(cli._FETCHES, "L2", lambda rest: seen.append(rest))
        cli.main(["fetch", "L2", "-u", "someone", "--dates", "20240405"])
        assert seen == [["-u", "someone", "--dates", "20240405"]]

    def test_return_value_propagates(self, monkeypatch):
        monkeypatch.setitem(cli._FETCHES, "L2", lambda rest: 7)
        assert cli.main(["fetch", "L2"]) == 7

    def test_every_subject_is_wired_to_its_own_script(self):
        assert cli._FETCHES == {
            "masters": fetch_masters.main,
            "L0": fetch_l0.main,
            "L2": fetch_l2.main,
            "L4": fetch_l4.main,
        }

    @pytest.mark.parametrize("argv", [["fetch"], ["fetch", "-h"]])
    def test_usage_returns_zero(self, capsys, argv):
        assert cli.main(argv) == 0
        assert "usage: kpfpipe fetch <subject>" in capsys.readouterr().out

    def test_unknown_subject_exits_two(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["fetch", "nonsense"])
        assert exc.value.code == 2
        assert "unknown subject 'nonsense'" in capsys.readouterr().err

    def test_fetch_is_listed_in_the_top_level_banner(self, capsys):
        cli.main([])
        assert "  fetch       " in capsys.readouterr().out


class TestUnknownCommand:
    def test_unknown_command_exits_two(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["frobnicate", "--foo"])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "unknown command" in err and "frobnicate" in err

    def test_plot_timeseries_is_not_a_command(self, capsys):
        # The plotter is a library the timeseries stage imports, not a driver; it
        # is deliberately not a CLI command and has no entry point of its own.
        assert "plot-timeseries" not in cli._COMMANDS
        with pytest.raises(SystemExit) as exc:
            cli.main(["plot-timeseries", "--target", "10700"])
        assert exc.value.code == 2
