"""Tests for scripts/processing/_argparse.py: the shared argparse parent parsers.

Each factory returns an ``add_help=False`` parent that the three subcommand
parsers compose via ``parents=[...]``. Drive each the way the commands do --
compose it into a throwaway parser and parse -- asserting the flags it
contributes, that data_dirs_parser gates ``--kpf_science_output`` on its argument,
and that pool_parser threads the caller's ``--jobs`` help text through.
"""

import argparse

import pytest

from scripts.processing import _argparse


def _parse(parents, argv):
    return argparse.ArgumentParser(parents=parents).parse_args(argv)


class TestRecipeParser:
    def test_adds_recipe_and_config(self):
        ns = _parse(
            [_argparse.recipe_and_config_parser()], ["-r", "/x.py", "-c", "/y.toml"]
        )
        assert ns.recipe == "/x.py"
        assert ns.config == "/y.toml"

    def test_long_options(self):
        ns = _parse(
            [_argparse.recipe_and_config_parser()],
            ["--recipe", "/x.py", "--config", "/y.toml"],
        )
        assert (ns.recipe, ns.config) == ("/x.py", "/y.toml")

    def test_default_none(self):
        ns = _parse([_argparse.recipe_and_config_parser()], [])
        assert ns.recipe is None and ns.config is None


class TestDataDirsParser:
    def test_includes_science_output_by_default(self):
        ns = _parse(
            [_argparse.data_dirs_parser()],
            ["--kpf_data_input", "/in", "--kpf_masters_output", "/m",
             "--kpf_science_output", "/s"],
        )  # fmt: skip
        assert ns.kpf_data_input == "/in"
        assert ns.kpf_masters_output == "/m"
        assert ns.kpf_science_output == "/s"

    def test_omits_science_output_when_disabled(self):
        parser = argparse.ArgumentParser(
            parents=[_argparse.data_dirs_parser(science_output=False)]
        )
        # The flag is absent from the namespace entirely...
        ns = parser.parse_args(["--kpf_data_input", "/in"])
        assert not hasattr(ns, "kpf_science_output")
        # ...and passing it is a parse error (masters produces no science output).
        with pytest.raises(SystemExit):
            parser.parse_args(["--kpf_science_output", "/s"])

    def test_input_dir_aliases_data_input(self):
        # --input_dir is a plain alias: it writes the kpf_data_input dest.
        ns = _parse([_argparse.data_dirs_parser()], ["--input_dir", "/in"])
        assert ns.kpf_data_input == "/in"

    def test_output_dir_parses_to_its_own_dest(self):
        ns = _parse([_argparse.data_dirs_parser()], ["--output_dir", "/out"])
        assert ns.output_dir == "/out"
        # It is a raw value here; the fan-out happens in resolve_dir_shortcuts.
        assert ns.kpf_masters_output is None and ns.kpf_science_output is None


class TestResolveDirShortcuts:
    def _parser(self, *, science_output=True, plot_dir=False):
        p = argparse.ArgumentParser(
            parents=[
                _argparse.data_dirs_parser(science_output=science_output),
                _argparse.logging_parser(),
            ]
        )
        if plot_dir:
            p.add_argument("--plot_dir", default=None)
        return p

    def test_fans_out_to_all_unset_slots(self):
        ns = _argparse.resolve_dir_shortcuts(
            self._parser(plot_dir=True).parse_args(["--output_dir", "/out"])
        )
        # masters/science outputs take the root; logs and plots get their subdirs.
        assert ns.kpf_masters_output == "/out"
        assert ns.kpf_science_output == "/out"
        assert ns.log_dir == "/out/logs"
        assert ns.plot_dir == "/out/QLP/timeseries"
        # The input dir is never touched by --output_dir.
        assert ns.kpf_data_input is None

    def test_explicit_flags_win(self):
        ns = _argparse.resolve_dir_shortcuts(
            self._parser(plot_dir=True).parse_args(
                [
                    "--output_dir",
                    "/out",
                    "--kpf_masters_output",
                    "/m",
                    "--plot_dir",
                    "/p",
                ]
            )
        )
        assert ns.kpf_masters_output == "/m"  # explicit kept
        assert ns.plot_dir == "/p"  # explicit kept
        # unset slots filled: science takes the root, log dir gets its subdir.
        assert ns.kpf_science_output == "/out" and ns.log_dir == "/out/logs"

    def test_skips_absent_slots(self):
        # masters-style parser: no science output, no plot dir -- no AttributeError.
        ns = _argparse.resolve_dir_shortcuts(
            self._parser(science_output=False).parse_args(["--output_dir", "/out"])
        )
        assert ns.kpf_masters_output == "/out" and ns.log_dir == "/out/logs"
        assert not hasattr(ns, "kpf_science_output")

    def test_noop_without_output_dir(self):
        ns = _argparse.resolve_dir_shortcuts(
            self._parser().parse_args(["--kpf_masters_output", "/m"])
        )
        assert ns.kpf_masters_output == "/m"
        assert ns.kpf_science_output is None and ns.log_dir is None


class TestLoggingParser:
    def test_adds_log_flags(self):
        ns = _parse(
            [_argparse.logging_parser()], ["--log_dir", "/l", "--log_level", "DEBUG"]
        )
        assert ns.log_dir == "/l" and ns.log_level == "DEBUG"


class TestPoolParser:
    def test_adds_jobs_and_timeout(self):
        ns = _parse(
            [_argparse.pool_parser(jobs_help="how many")],
            ["--jobs", "4", "--job_timeout", "120"],
        )
        assert ns.jobs == 4 and ns.job_timeout == 120

    def test_defaults(self):
        ns = _parse([_argparse.pool_parser(jobs_help="how many")], [])
        assert ns.jobs is None and ns.job_timeout == 600

    def test_jobs_help_is_passed_through(self):
        parser = argparse.ArgumentParser(
            parents=[_argparse.pool_parser(jobs_help="SENTINEL help text")]
        )
        assert "SENTINEL help text" in parser.format_help()
