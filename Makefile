.PHONY: notebook test test-fast test-serial

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
