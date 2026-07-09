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
        ns = _parse([_argparse.recipe_parser()], ["-r", "/x.py", "-c", "/y.toml"])
        assert ns.recipe == "/x.py"
        assert ns.config == "/y.toml"

    def test_long_options(self):
        ns = _parse(
            [_argparse.recipe_parser()], ["--recipe", "/x.py", "--config", "/y.toml"]
        )
        assert (ns.recipe, ns.config) == ("/x.py", "/y.toml")

    def test_default_none(self):
        ns = _parse([_argparse.recipe_parser()], [])
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
