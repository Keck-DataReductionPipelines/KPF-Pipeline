.PHONY: notebook test test-fast test-cli test-qlp test-serial test-debug profile profile-science profile-masters

notebook:
	jupyter notebook --port ${KPFPIPE_PORT} --allow-root --ip=0.0.0.0 --no-browser

# All targets run through `conda run -n kpfpipe` — base-system Python lacks
# rvdata and fails with ModuleNotFoundError.

# Full suite, parallel (pytest-xdist). loadscope keeps each class on one worker
# so the class-scoped integration fixtures run once, not once per worker.
# Run before a PR, after core/shared-module changes, or after major refactors.
test:
	conda run -n kpfpipe python -m pytest tests/ -n auto --dist loadscope

# Fast pre-commit subset: recipes and below. Excludes @pytest.mark.slow
# (integration + heavy-compute), @pytest.mark.cli (the scripts/CLI/tools layer),
# and @pytest.mark.quicklook (slow QLP PNG rendering, off the production path).
# The default during day-to-day development.
test-fast:
	conda run -n kpfpipe python -m pytest tests/ -m "not slow and not cli and not quicklook" -n auto --dist loadscope

# scripts/CLI/tools layer only (@pytest.mark.cli) -- run when working on
# scripts/ or tools/ directly. The full `make test` covers this too.
test-cli:
	conda run -n kpfpipe python -m pytest tests/ -m cli -n auto --dist loadscope

# Quicklook/QLP render suite only (@pytest.mark.quicklook) -- run when working on
# the quicklook plots directly. The full `make test` covers this too.
test-qlp:
	conda run -n kpfpipe python -m pytest tests/ -m quicklook -n auto --dist loadscope

# Serial fallback (no xdist) — for debugging parallel/receipt issues.
test-serial:
	conda run -n kpfpipe python -m pytest tests/

# Debug loop: rerun only last-failed (--lf) and halt at the first failure (-x).
# Run a full target once to seed the failure cache, then iterate on this.
test-debug:
	conda run -n kpfpipe python -m pytest --lf -x

# ---------------------------------------------------------------------------
# Profiling ("tallest tentpole" suite). Standalone harnesses in tests/profiling/
# (not named test_*.py, so pytest never collects them and `make test` stays fast).
# They run on the real frames in tests/testdata and skip cleanly when absent.
# Auto-generated reports land in tests/profiling/reports/ (gitignored). See the
# "## Profiling" section of CLAUDE.md.
# ---------------------------------------------------------------------------

# Per-module: `make profile-radial_velocity`, `make profile-master_wls`, etc.
# These names mirror the test files 1-to-1 (test_<x>.py <-> profile_<x>.py).
PROFILE_MODULES = image_assembly image_processing spectral_extraction \
	wavelength_calibration barycentric_correction cross_correlation \
	radial_velocity calibration_association master_bias master_dark order_trace \
	master_wls

# Run every harness (both recipes + all per-module files) and regenerate reports.
profile: profile-masters profile-science $(addprefix profile-,$(PROFILE_MODULES))

# End-to-end recipes (ranked across modules to find the dominant stage).
profile-science:
	conda run -n kpfpipe python -m tests.profiling.profile_science_recipe

profile-masters:
	conda run -n kpfpipe python -m tests.profiling.profile_masters_recipe

# Generic per-module rule: profile-<module> -> tests/profiling/profile_<module>.py
profile-%:
	conda run -n kpfpipe python -m tests.profiling.profile_$*
