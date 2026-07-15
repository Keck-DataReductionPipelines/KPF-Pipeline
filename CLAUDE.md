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
# Run from KPF-Pipeline/ (git receipt system requirement).

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

# Profiling harnesses (design: architecture guide → Tests → Profiling)
make profile   # all harnesses; also profile-science / profile-masters / profile-<module>
```

## Running tests — subset vs full

Which *tier* to run, not how often: testing is a judgment call made continuously
while working, not deferred to commit/PR time (no git hook runs tests — `pre-commit`
is ruff-only). Match the scope to the change instead of defaulting to the full suite:

- **While iterating (most runs):** only the file(s) for the code you touched
  (`python -m pytest tests/regression/test_<area>.py`).
- **Before wrapping up / committing:** `make test-fast` (the `-m "not slow"` subset, ~16s).
- **The full suite (`make test`)** — when the blast radius is wide: opening/updating a
  PR; a change to a **core/shared** module other tests depend on (the data models,
  `kpfpipe/__init__.py` constants, base classes, anything the integration tests exercise);
  or a cross-cutting refactor.

The fast subset skips the `slow` integration tests (real-frame assembly/overscan, master
stacking, full L0→L2, WLS orientation), which is why the full-suite triggers above are what
catch those. How the suite is laid out and split — and the masters test layout — is in the
architecture guide (*Tests → Regression*); test-writing conventions are in the style guide §12.

## Architecture

The pipeline architecture — the data-model hierarchy, extension alias system, data flow,
header standardization, configuration, the CLI/module layering, logging, filename
conventions, the masters pipeline, the diagnostics/QC/checkpoint/quicklook layers, and the
test/profiling suite structure — lives in
[`KPF_VNEXT_ARCHITECTURE.md`](docs/dev/KPF_VNEXT_ARCHITECTURE.md) (governing doc #4).
It is not duplicated here; consult it before making structural or cross-cutting changes.

## Design Principles & Success Criteria

These live in the charter and are NOT restated here (the two copies drifted in the past — keep one source). See [`KPF_VNEXT_CHARTER.md`](docs/dev/KPF_VNEXT_CHARTER.md): §10 Core Design Principles, §9 Guardrails, §5 Calibration Philosophy, §3 Definition of Success, §6 (every major change must preserve deterministic behavior, run on the truth dataset, and document impact on RV metrics). Consult the charter before design decisions.

- Keep this file (CLAUDE.md) updated with operational lessons, conventions, and more efficient workflows learned while coding.
- Use CLAUDE.md as long-term memory for technical/operational guidance; use the charter for project intent and principles.
