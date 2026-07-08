"""Tests for tools/cli.py: the ``kpfpipe`` subcommand dispatcher.

tools/cli.py is a thin, git-style router: it maps the first argument to a
subcommand implementation under ``scripts/processing/`` and forwards the
remaining argv verbatim. These tests verify the routing (each command reaches
its handler with the untouched remainder), the usage banner, and the
unknown-command error -- the subcommands' own parsing is tested in
test_reduce.py / test_masters_script.py / test_science_script.py.
"""

import pytest

from tools import cli


class TestDispatch:
    @pytest.mark.parametrize("command", ["run", "masters", "science"])
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

    @pytest.mark.parametrize("flag", ["-h", "--help"])
    def test_help_flag_prints_usage(self, flag, capsys):
        assert cli.main([flag]) == 0
        assert "usage: kpfpipe <command>" in capsys.readouterr().out


class TestUnknownCommand:
    def test_unknown_command_exits_two(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["frobnicate", "--foo"])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "unknown command" in err and "frobnicate" in err
