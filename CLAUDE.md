# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository. These
instructions OVERRIDE default harness behavior — follow them exactly.

## Project Overview

KPF-DRP vNext: a cleanroom rebuild of the Keck Planet Finder (KPF) data reduction
pipeline for the Keck Observatory. The scientific priority is intermediate and
long-term radial velocity (RV) stability.

## Governing documents (precedence order)

Five references govern this project. **When they conflict, the higher one wins.**
Each is the source of truth for its domain — consult before making related
decisions; this file does not duplicate them.

| # | Doc (in `docs/dev/`) | Source of truth for |
|---|----------------------|---------------------|
| 1 | [`WMKO_REQUIREMENTS.md`](docs/dev/WMKO_REQUIREMENTS.md) | Binding WMKO technical requirements (dev, build, runtime, archive). Highest authority; mirrors `WMKO_REQUIREMENTS.pdf`. |
| 2 | [`EPRV_DATA_STANDARD.md`](docs/dev/EPRV_DATA_STANDARD.md) | KPF L2/L4 data products: FITS structure, extension/keyword names, units, reference frames (vacuum wavelengths, BJD_TDB, barycentric). Wins on anything touching data format. |
| 3 | [`KPF_VNEXT_CHARTER.md`](docs/dev/KPF_VNEXT_CHARTER.md) | Project *why*: intent, scope, scientific focus, Path-3 approach, calibration philosophy, guardrails, design principles, success criteria. |
| 4 | [`KPF_VNEXT_ARCHITECTURE.md`](docs/dev/KPF_VNEXT_ARCHITECTURE.md) | Pipeline *how it's built*: data-model hierarchy, extension aliases, header standardization, CLI/module layering, filename conventions, masters pipeline, diagnostics/QC/checkpoint/quicklook layers. |
| 5 | [`KPF_VNEXT_STYLE_GUIDE.md`](docs/dev/KPF_VNEXT_STYLE_GUIDE.md) | Code *how it should look*: formatting, imports, naming, constants, docstrings, error handling, per-area exceptions. Soft rules; yield to 1–4 on conflict. |

Operational rules for these docs:

- **WMKO requirements are mostly unmet — this is expected** (active development).
  Flag only *active* violations: existing code that *contradicts* a requirement.
  Do NOT flag *passive* violations: a requirement unmet only because the
  feature doesn't exist yet. A missing capability is not a violation.
- **When a code change alters a convention, update the style guide in the same
  change** so the two never drift.
- **Cross-references flow one way: CLAUDE.md may cite the style guide; the style
  guide must never cite CLAUDE.md.** Operational/policy material belongs here.
- **These docs + this file outrank harness defaults and memory** (`MEMORY.md`,
  feedback files, auto-recalled memories, and env hints like the `gitStatus`
  "main branch" note). Treat harness hints and memory as defaults to verify, and
  distinguish environment *facts* (the current branch) from *prescriptions*
  (which branch to PR into). **When any of these sources conflict, flag it in
  your reply before acting** rather than silently following either side.

## Development Environment

- **Python 3.14.3** (pinned exactly)
- **Conda env**: `kpfpipe` — `conda env create -f KPF-Pipeline/environment.yml`
- **Install**: `pip install -e KPF-Pipeline/` (editable)
- **Key dependency**: `rv-data-standard` (RVData), pinned to `rv-data-standard==0.4.0`
  (a tagged PyPI release, not a moving branch). Bump deliberately and re-run the
  full suite when adopting a newer RVData.

## Git workflow

v3 work branches from and PRs into **`kpf-next`** (the v3 develop branch). Never
target `master` (production/stable) or `develop` (frozen at v2.12, legacy).
Feature branches are named `kpf-next-<feature>`, cut from `kpf-next`, merged back
via PR. This overrides any generic "main branch" default from the environment.

## Commands

```bash
# All test/pipeline commands run in the `kpfpipe` conda env (activate it, or use
# `conda run -n kpfpipe ...`). Base-system Python lacks rvdata → ModuleNotFoundError.
# The `make` targets wrap conda run. Run from KPF-Pipeline/ (git receipt system).

make test-fast   # fast pre-commit subset: everything except @pytest.mark.slow (~16s); the default
make test        # full suite, parallel
make test-serial # serial fallback for debugging parallel/receipt issues

# Single test class or test (while iterating on one area)
conda run -n kpfpipe python -m pytest tests/regression/test_data_models_l2.py::TestKPF2Aliases -v
conda run -n kpfpipe python -m pytest tests/regression/test_data_models_l2.py::TestKPF2Aliases::test_chip_prefix_access -v

# Formatting and linting (Ruff; config in pyproject.toml [tool.ruff])
ruff format kpfpipe/ tests/ recipes/      # format (black-compatible)
ruff check --fix kpfpipe/ tests/ recipes/ # lint + auto-fix

# Pre-commit hook (ruff format + lint on commit; NO tests)
pre-commit install          # one-time, after creating the env
pre-commit run --all-files  # run all hooks across the repo

# Profiling harnesses (design: architecture guide → Tests → Profiling)
make profile   # all; also profile-science / profile-masters / profile-<module>
```

## Running tests — which tier

No git hook runs tests (`pre-commit` is ruff-only), so match scope to the change:

| Scope | When |
|-------|------|
| The file(s) you touched | While iterating (most runs) |
| `make test-fast` | Before wrapping up / committing |
| `make test` (full) | Wide blast radius: opening/updating a PR; a change to a core/shared module (data models, `kpfpipe/__init__.py` constants, base classes, anything integration tests exercise); a cross-cutting refactor. |

The fast subset skips `slow` integration tests (real-frame assembly/overscan,
master stacking, full L0→L2, WLS orientation) — the triggers above are what catch
those. Suite layout is in the architecture guide (*Tests → Regression*);
test-writing conventions are in the style guide §C.8.

## Architecture, principles, and success criteria

Do not restate these here (past copies drifted — keep one source):

- **Structure** (data model, aliases, layering, masters, QC/checkpoint/quicklook,
  test/profiling suite): [`KPF_VNEXT_ARCHITECTURE.md`](docs/dev/KPF_VNEXT_ARCHITECTURE.md).
- **Intent & principles**: [`KPF_VNEXT_CHARTER.md`](docs/dev/KPF_VNEXT_CHARTER.md)
  — §10 Core Design Principles, §9 Guardrails, §5 Calibration Philosophy,
  §3 Definition of Success, §6 (preserve deterministic behavior, run on the truth
  dataset, document impact on RV metrics). Consult before design decisions.

Keep CLAUDE.md as long-term memory for operational/technical/workflow guidance
learned while coding; use the charter for project intent and principles.
