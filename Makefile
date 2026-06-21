.PHONY: notebook test test-fast test-serial profile profile-science profile-masters

notebook:
	jupyter notebook --port ${KPFPIPE_PORT} --allow-root --ip=0.0.0.0 --no-browser

# All targets run through `conda run -n kpfpipe` — base-system Python lacks
# rvdata and fails with ModuleNotFoundError.

# Full suite, parallel (pytest-xdist). loadscope keeps each class on one worker
# so the class-scoped integration fixtures run once, not once per worker.
# Run before a PR, after core/shared-module changes, or after major refactors.
test:
	conda run -n kpfpipe python -m pytest tests/ -n auto --dist loadscope

# Fast pre-commit subset: everything except @pytest.mark.slow (integration +
# heavy-compute). The default during day-to-day development.
test-fast:
	conda run -n kpfpipe python -m pytest tests/ -m "not slow" -n auto --dist loadscope

# Serial fallback (no xdist) — for debugging parallel/receipt issues.
test-serial:
	conda run -n kpfpipe python -m pytest tests/

# ---------------------------------------------------------------------------
# Profiling ("tallest tentpole" suite). Standalone harnesses in tests/ that are
# NOT named test_*.py, so pytest never collects them and `make test` stays fast.
# They run on the real frames in tests/testdata and skip cleanly when absent.
# Auto-generated reports land in tests/profiling/reports/ (gitignored). See the
# "## Profiling" section of CLAUDE.md; curated analysis lives in PROFILING.md.
# ---------------------------------------------------------------------------

# Per-module: `make profile-radial_velocity`, `make profile-master_wls`, etc.
# These names mirror the test files 1-to-1 (test_<x>.py <-> profile_<x>.py).
PROFILE_MODULES = image_assembly image_processing spectral_extraction \
	wavelength_calibration barycentric_correction radial_velocity \
	calibration_association master_base master_bias master_dark master_wls

# Run every harness (both recipes + all per-module files) and regenerate reports.
profile: profile-masters profile-science $(addprefix profile-,$(PROFILE_MODULES))

# End-to-end recipes (ranked across modules to find the dominant stage).
profile-science:
	conda run -n kpfpipe python -m tests.profile_science_recipe

profile-masters:
	conda run -n kpfpipe python -m tests.profile_masters_recipe

# Generic per-module rule: profile-<module> -> tests/profile_<module>.py
profile-%:
	conda run -n kpfpipe python -m tests.profile_$*
