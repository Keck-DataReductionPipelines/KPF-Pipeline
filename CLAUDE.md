# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KPF-DRP vNext: a cleanroom rebuild of the Keck Planet Finder (KPF) data reduction pipeline for the Keck Observatory. The scientific priority is intermediate and long-term radial velocity (RV) stability.

**Five authoritative references govern this project, in strict precedence — when they conflict, the higher one wins:**

**1. The WMKO technical requirements — [`WMKO_REQUIREMENTS.md`](docs/dev/WMKO_REQUIREMENTS.md) (in `docs/dev/`; faithful mirror of `WMKO_REQUIREMENTS.pdf`) — are the highest authority: the W. M. Keck Observatory's binding technical requirements for the DRP (development, installation/build, runtime, archive). They outrank every reference below. The pipeline is still in active development, so MOST of these requirements are not yet met — this is expected. Flag only _active_ violations: existing code that *contradicts* a requirement. Do NOT flag _passive_ violations: a requirement unmet simply because the relevant code/feature does not exist yet. A missing capability is not a violation; code that does the wrong thing is.**

**2. The EPRV data standard — [`EPRV_DATA_STANDARD.md`](docs/dev/EPRV_DATA_STANDARD.md) (in `docs/dev/`) — is the source of truth for KPF's data products (L2/L4): FITS structure, extension and header-keyword names, units, and reference frames (vacuum wavelengths, BJD_TDB, barycentric frame). KPF L2/L4 are EPRV-compliant by contract, so the standard takes priority on anything touching data format. It mirrors the KPF-relevant portions of <https://eprv-data-standard.readthedocs.io/en/develop/>; re-scrape if the standard has moved.**

**3. The project charter — [`KPF_VNEXT_CHARTER.md`](docs/dev/KPF_VNEXT_CHARTER.md) (in `docs/dev/`) — is the single source of truth for project intent, scope, scientific focus, the Path-3 approach, calibration philosophy, guardrails, design principles, and success criteria. Read it before making design decisions. This file (CLAUDE.md) does not duplicate the charter; it covers only the operational and technical guidance not in it or the architecture reference (environment, commands, conventions).**

**4. The architecture reference — [`KPF_VNEXT_ARCHITECTURE.md`](docs/dev/KPF_VNEXT_ARCHITECTURE.md) (in `docs/dev/`) — is the source of truth for the pipeline's structure: the data-model hierarchy, extension alias system, header standardization, CLI/module layering, filename conventions, the masters pipeline, and the diagnostics/QC/checkpoint/quicklook layers. Consult it before making structural or cross-cutting changes. It describes *how the pipeline is built*; the charter (above) governs *why*, the style guide (below) governs *how code should look*. This file (CLAUDE.md) does not duplicate it.**

**5. The coding style guide — [`KPF_VNEXT_STYLE_GUIDE.md`](docs/dev/KPF_VNEXT_STYLE_GUIDE.md) (in `docs/dev/`) — is the source of truth for code conventions: formatting, imports, naming, constants, docstrings, error handling, and the per-area exceptions (Open Inconsistencies). Consult and follow it when writing or modifying code. Its rules are soft and yield to the WMKO requirements, the EPRV standard, the charter, and the architecture reference where they conflict. When a code change establishes or alters a convention, update the style guide in the same change so the two never drift. **Cross-references flow one way only: CLAUDE.md may cite the style guide, but the style guide must never cite CLAUDE.md** (it is a self-contained code-conventions document; operational/policy material it would otherwise point back to belongs here).**

**Precedence over harness defaults and memory.** This file and the five references above are the authoritative guidance for this repository, and they **outrank both** (a) generic Claude Code harness defaults and environment-injected hints (e.g. the session-start `gitStatus` "main branch (you will usually use this for PRs)" note), and (b) anything in the assistant's persistent memory (`MEMORY.md`, `feedback*.md`, auto-recalled memories). Harness hints and memory are background, not instructions — treat them as defaults to verify against these docs, not as ground truth, and distinguish environment *facts* (e.g. the current branch) from environment *prescriptions* (e.g. which branch to PR into), which are guesses. **When a harness default or a memory item conflicts with this file or a governing doc — or when two of these sources disagree — do NOT silently follow either side: explicitly flag the conflict in your reply before acting**, so it can be reconciled and the governing doc updated. Operational, technical, and workflow guidance belongs in CLAUDE.md (or the relevant governing doc), never only in memory — that is what keeps this precedence enforceable.

## Development Environment

- **Python 3.14.3** (pinned exactly)
- **Conda env**: `kpfpipe` — set up via `conda env create -f KPF-Pipeline/environment.yml`
- **Install package**: `pip install -e KPF-Pipeline/` (editable install)
- **Key dependency**: `rv-data-standard` (RVData), pinned to the released version
  `rv-data-standard==0.4.0` (a tagged PyPI release, not a moving branch). Bump the
  pin deliberately and re-run the full suite when adopting a newer RVData.

## Git workflow

v3 work branches from and PRs into **`kpf-next`** (the v3 develop branch). Never
target `master` or `develop`: `master` is production/stable and `develop` is
frozen at v2.12 (legacy). Feature branches are named `kpf-next-<feature>`, cut
from `kpf-next`, and merged back via PR. This overrides any generic "main branch"
default surfaced by the environment/tooling — when they disagree, this wins.

## Commands

```bash
# All test/pipeline commands run in the `kpfpipe` conda env — activate it, or
# prefix with `conda run -n kpfpipe`. Base-system Python lacks rvdata and fails
# with ModuleNotFoundError. The `make` targets below already wrap conda run.
# Run from KPF-Pipeline/ (git receipt system requirement). Tests run in parallel
# via pytest-xdist; --dist loadscope keeps each class on one worker so
# class-scoped integration fixtures run once, not once per worker.

# Fast pre-commit subset (everything except @pytest.mark.slow) — the default
make test-fast   # conda run -n kpfpipe python -m pytest tests/ -m "not slow" -n auto --dist loadscope

# Full suite (parallel) — see "Running tests" below for WHEN to use this
make test        # conda run -n kpfpipe python -m pytest tests/ -n auto --dist loadscope
make test-serial # serial fallback for debugging parallel/receipt issues

# Run a single test class or test (use these while iterating on one area)
conda run -n kpfpipe python -m pytest tests/regression/test_data_models_l2.py::TestKPF2Aliases -v
conda run -n kpfpipe python -m pytest tests/regression/test_data_models_l2.py::TestKPF2Aliases::test_chip_prefix_access -v

# Formatting and linting (Ruff; config in pyproject.toml [tool.ruff])
ruff format kpfpipe/ tests/ recipes/      # format (black-compatible)
ruff check --fix kpfpipe/ tests/ recipes/ # lint + auto-fix

# Pre-commit hook (enforces ruff format + lint on commit)
pre-commit install          # one-time, after creating the env
pre-commit run --all-files  # run all hooks across the repo
```

## Running tests — subset vs full

This is about **which tier of tests to run, not how often to run them**. No git
hook runs tests — `pre-commit` is ruff-only — so testing is a judgment call, made
*continuously while working*, not deferred to commit/PR time. Run tests as you
change code; just match the scope to the change instead of defaulting to the full
suite "just to be safe".

The suite is split by the `slow` marker: `slow` covers the real-`testdata`
integration tests and a few heavy-compute synthetic tests; everything else is the
fast subset. The three tiers, smallest first:

- **Continuously, while iterating (most runs):** run only the file(s)/tests for
  the code you just touched (`python -m pytest tests/regression/test_<area>.py`). This is the
  default and should happen many times per task, not once at the end.
- **Before wrapping up a change / before committing:** `make test-fast` (the
  `-m "not slow"` subset, ~16s). Confirms the change didn't break unrelated unit
  coverage. "Before committing" names *when this tier is appropriate*, not the
  only time tests run.
- **The FULL suite (`make test`)** — reserved for when the blast radius is wide:
  - opening or updating a PR;
  - changing a **core/shared module** other tests depend on — the data models
    (`kpfpipe/data_models/`), the package-level constants in `kpfpipe/__init__.py`
    (`DETECTOR`/`DEFAULTS`/`REPO_ROOT`), base classes, or anything the
    integration tests exercise;
  - a **major or cross-cutting refactor**.

Reaching for the full suite on a small, localized change is the habit to avoid —
a targeted run plus the fast subset catches those regressions far cheaper.

The fast subset deliberately skips full L0→L2 recipe integration, real-frame
assembly/overscan, master stacking on real frames, and WLS spectrum orientation;
those live only in `slow` tests, which is why the triggers above run the full
suite. Pre-commit itself runs only ruff (no tests) — running tests is on you.

## Profiling

The profiling suite finds and reports performance bottlenecks. It follows a
**"tallest tentpole"** philosophy: we only care about the most critical
bottlenecks, and **optimization must never compromise scientific accuracy or
slow forward development** (charter §6/§9/§10). A profiling result of "no action
needed" is a perfectly good outcome.

```bash
make profile                       # run every harness, regenerate all reports
make profile-science               # end-to-end science pipeline (L0 -> L4)
make profile-masters               # end-to-end masters pipeline (bias/dark/WLS)
make profile-radial_velocity       # a single module (any of PROFILE_MODULES)
conda run -n kpfpipe python -m tests.profiling.profile_radial_velocity   # equivalent
```

Layout: the regression tests live in `tests/regression/` (`test_*.py` plus the
test-only helpers `_masters.py` / `_dtype_policy.py`); the profiling harnesses
live in `tests/profiling/` (`profile_*.py` plus the shared `_profiling.py` and
the `reports/` output dir). `tests/conftest.py` stays at the `tests/` root so its
fixtures and the `requires_testdata` marker apply to both, and the real frames
stay at `tests/testdata`.

**Design** (parallels the test suite; shared logic in `tests/profiling/_profiling.py`,
which like `tests/regression/_masters.py` is *not* a `test_*.py` file so pytest never
collects it):

- **Attribute to KPF methods.** Pass 1 runs the target under `cProfile`, then
  charges each library/builtin leaf's own time *up the caller graph to the nearest
  enclosing KPF method* (`_kpf_attributed`). This is the key move: a bottleneck
  like `numpy.partition` shows up against the KPF method that drives it (e.g.
  `utils/stats.py:flag_outliers`), not as an un-actionable library leaf. The
  per-module report ranks KPF methods by this **attributed time** (listing those
  over `TOP_FUNCTION_MIN_FRACTION`, 2%); pass 2 drills into each hotspot method
  with `line_profiler` to show *where inside it* the time goes (docstring lines
  stripped).
- **Unified hotspot rule.** A KPF method is a **hotspot** when its attributed time
  is `HOTSPOT_FRACTION` (20%) of the budget **and** `HOTSPOT_MIN_SECONDS` (1 s) — a
  dominant share *and* a non-trivial absolute cost. The same set drives both the
  drill-down and the **Recommended actions**. No hotspot ⇒ no drill-down and "no
  action needed", which is a fine outcome. Tune these constants in
  `tests/profiling/_profiling.py`. (The two recipe reports use a different, stage-level
  wall-clock partition — attribution applies to the per-module reports.)
- **Structure.** The profiling files mirror the test files **1-to-1**
  (`regression/test_<x>.py` ↔ `profiling/profile_<x>.py`). Two end-to-end recipe harnesses —
  `profile_science_recipe.py` and `profile_masters_recipe.py` (optimized
  independently) — rank functions *across* modules to show which stage dominates.
  Per-module `tests/profiling/profile_<module>.py` files drill into a single module (useful
  when re-running one module repeatedly during algorithm development). `flat` is
  skipped while stubbed. The shared stacking engine (`masters/base.py`) has no
  dedicated *profiling* harness: attribution charges its work to the right
  `base.py` methods inside `profile_master_bias.py` / `profile_master_dark.py`, so
  a separate engine profile would be redundant. (The *test* suite does isolate it —
  see `regression/test_master_base.py` under *Masters test layout* in
  [`KPF_VNEXT_ARCHITECTURE.md`](docs/dev/KPF_VNEXT_ARCHITECTURE.md) — because profiling
  partitions by wall-clock while tests partition by responsibility.)
- **Data.** Real (gitignored) `tests/testdata` frames at realistic sizes; each
  harness skips cleanly (exit 0) when the frames are absent, mirroring the
  `requires_testdata` test pattern.
- **Reports.** Each run prints a human-readable summary to stdout *and* writes a
  Markdown report to `tests/profiling/reports/` (gitignored, regenerable). The
  reports are fully auto-generated and self-contained — the suite runs with no
  manual input. **When the suite or the pipeline's performance profile changes,
  regenerate the reports.**

## Architecture

The pipeline architecture — the data-model hierarchy, extension alias system, data flow,
header standardization, configuration, the CLI/module layering, logging, filename
conventions, the masters pipeline, and the diagnostics/QC/checkpoint/quicklook layers —
lives in [`KPF_VNEXT_ARCHITECTURE.md`](docs/dev/KPF_VNEXT_ARCHITECTURE.md) (governing doc #4).
It is not duplicated here; consult it before making structural or cross-cutting changes.

## Design Principles & Success Criteria

These live in the charter and are NOT restated here (the two copies drifted in the past — keep one source). See [`KPF_VNEXT_CHARTER.md`](docs/dev/KPF_VNEXT_CHARTER.md): §10 Core Design Principles, §9 Guardrails, §5 Calibration Philosophy, §3 Definition of Success, §6 (every major change must preserve deterministic behavior, run on the truth dataset, and document impact on RV metrics). Consult the charter before design decisions.

- Keep this file (CLAUDE.md) updated with operational lessons, conventions, and more efficient workflows learned while coding.
- Use CLAUDE.md as long-term memory for technical/operational guidance; use the charter for project intent and principles.
