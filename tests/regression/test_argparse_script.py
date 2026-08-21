"""Tests for scripts/processing/_argparse.py: the shared argparse parent parsers.

Each factory returns an ``add_help=False`` parent that the subcommand parsers
compose via ``parents=[...]``, so each is driven the way the commands drive it:
composed into a throwaway parser, then parsed.
"""

import argparse

import pytest

from scripts.processing import _argparse

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli


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
        # Masters produces no science output, so the flag is neither set nor accepted.
        ns = parser.parse_args(["--kpf_data_input", "/in"])
        assert not hasattr(ns, "kpf_science_output")
        with pytest.raises(SystemExit):
            parser.parse_args(["--kpf_science_output", "/s"])

    def test_input_dir_aliases_data_input(self):
        ns = _parse([_argparse.data_dirs_parser()], ["--input_dir", "/in"])
        assert ns.kpf_data_input == "/in"

    def test_output_dir_parses_to_its_own_dest(self):
        ns = _parse([_argparse.data_dirs_parser()], ["--output_dir", "/out"])
        assert ns.output_dir == "/out"
        # Still a raw value here; the fan-out happens in resolve_dir_shortcuts.
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
        assert ns.kpf_masters_output == "/out"
        assert ns.kpf_science_output == "/out"
        assert ns.log_dir == "/out/logs"
        assert ns.plot_dir == "/out/QLP/timeseries"
        assert ns.kpf_data_input is None  # --output_dir never touches the input dir

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
        assert ns.kpf_masters_output == "/m"
        assert ns.plot_dir == "/p"
        assert ns.kpf_science_output == "/out" and ns.log_dir == "/out/logs"

    def test_skips_absent_slots(self):
        # A masters-style parser has no science output or plot dir: absent slots
        # must be skipped, not raise AttributeError.
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
        assert ns.jobs is None and ns.job_timeout == 1200

    def test_jobs_help_is_passed_through(self):
        parser = argparse.ArgumentParser(
            parents=[_argparse.pool_parser(jobs_help="SENTINEL help text")]
        )
        assert "SENTINEL help text" in parser.format_help()


class TestCacheParser:
    """The --cache mode flag. Its parameterised default is what makes the leaf
    `reduce` read-only while the orchestrators own cache writing -- the
    one-writer-per-cache-file invariant documented at _scan.py:1-15."""

    def test_factory_default_is_read_only(self):
        assert _parse([_argparse.cache_parser()], []).cache == "r"

    def test_orchestrator_default_is_honoured(self):
        assert _parse([_argparse.cache_parser(default="rw")], []).cache == "rw"

    @pytest.mark.parametrize("mode", ["r", "w", "rw", "wr"])
    def test_every_choice_parses(self, mode):
        assert _parse([_argparse.cache_parser()], ["--cache", mode]).cache == mode

    def test_unknown_mode_rejected(self):
        with pytest.raises(SystemExit):
            _parse([_argparse.cache_parser()], ["--cache", "rr"])
