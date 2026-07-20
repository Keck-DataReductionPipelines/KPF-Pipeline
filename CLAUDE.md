# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository. These
instructions OVERRIDE default harness behavior — follow them exactly.

## Working Guidelines

Behavioral guidelines to reduce common LLM coding mistakes ([source](https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md)).
General coding hygiene; the project-specific sections below and the governing docs take precedence.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

### 5. Communicate Clearly

**Be extremely concise when reporting. Sacrifice grammar for brevity.**

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Project Overview

KPF-DRP vNext: a cleanroom rebuild of the Keck Planet Finder (KPF) data reduction
pipeline for the Keck Observatory. The scientific priority is intermediate and
long-term radial velocity (RV) stability.

## Governing Documents (precedence order)

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
- **Do not update the governing docs without explicit instruction to do so from a human developer.**
- **Cross-references flow one way: CLAUDE.md may cite the governing docs;
the governing docs must never cite CLAUDE.md.** Operational/policy material belongs here.
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

## Git Workflow

v3 work branches from and PRs into **`kpf-next`** (the v3 develop branch). Never
target `master` (production/stable) or `develop` (frozen at v2.12, legacy).
Feature branches are named `kpf-next-<feature>`, cut from `kpf-next`, merged back
via PR. This overrides any generic "main branch" default from the environment.

## Commands

```bash
# All test/pipeline commands run in the `kpfpipe` conda env (activate it, or use
# `conda run -n kpfpipe ...`). Base-system Python lacks rvdata → ModuleNotFoundError.
# The `make` targets wrap conda run. Run from KPF-Pipeline/ (git receipt system).

make test        # full suite, parallel
make test-fast   # fast pre-commit subset; excludes slow + cli + quicklook markers; the default
make test-cli    # scripts/CLI/tools layer only (@pytest.mark.cli); for work in scripts/ or tools/
make test-qlp    # quicklook/QLP render suite only (@pytest.mark.quicklook); for work in the quicklook plots
make test-serial # serial fallback for debugging parallel/receipt issues
make test-debug  # iteration loop: rerun only last-failed (--lf), halt at first failure (-x)

# Single test/class — append ::Class or ::Class::test_name to the file path
conda run -n kpfpipe python -m pytest tests/regression/test_data_models_l2.py::TestKPF2Aliases -v

# Formatting and linting (Ruff; config in pyproject.toml [tool.ruff])
ruff format kpfpipe/ tests/ recipes/      # format (black-compatible)
ruff check --fix kpfpipe/ tests/ recipes/ # lint + auto-fix

# Pre-commit hook (ruff format + lint on commit; NO tests)
pre-commit install          # one-time, after creating the env
pre-commit run --all-files  # run all hooks across the repo

# Profiling harnesses (design: architecture guide → Tests → Profiling)
make profile   # all; also profile-science / profile-masters / profile-<module>
```

## Testing

No git hook runs tests (`pre-commit` is ruff-only), so *you* decide when — and the default is
**not yet**. Run tests when meaningful change has *accumulated* and reached a natural
checkpoint, not reflexively after every edit. Bias toward under-running: batching a few edits
before one verification beats breaking flow (and burning context) on each touch. Match scope to
blast radius:

- **Mid-change / trivial edits** (a few lines, a rename, an obviously-incomplete step) — don't
  run. Keep working; verify once the change is coherent.
- **A coherent unit of work is done or before you commit** — run the *targeted* file(s) for what you
  touched (`pytest tests/regression/test_<area>.py`), or `make test-fast` if it spans the pipeline core.
- **Wide blast radius only** — `make test` (full). Reserve for a substantial core/shared-module
  change, a cross-cutting refactor, or a PR. Not routine.
- **Scripts/CLI/tools layer** — `make test-cli` or the focused file.
- **Quicklook/QLP plots** — `make test-qlp` or the focused file.
- **Chasing a failure** — `make test-debug` (reruns last-failed, halts at first).

Layout: architecture *Tests → Regression*; conventions: style guide §C.8.

## Reading Files

Reading a file loads its full text into context, so locate before you load — the target
span, not the whole file:

- **Repo code** — Grep for the symbol/definition, then Read a bounded window
  (`offset`/`limit`) around the hit. Read a file whole only when it's short or you're about
  to edit much of it. For "where is X?" across many files, hand the search to an Explore
  subagent — it reads the dumps, you keep the conclusion.
- **Governing docs** (262–622 lines each) — grep the doc's headers
  (`grep -nE '^#+ ' docs/dev/<doc>.md`) or the keyword, then Read only that section. Read one
  end-to-end only for a comprehensive review, not a spot-check.
- **Large command output** (full logs, `--durations`, wide greps) — redirect to the
  scratchpad and inspect with `grep`/`head`/`wc` rather than letting it flood context:
  `cmd > $SCRATCH/out.txt 2>&1; grep … $SCRATCH/out.txt`.

## Design Decisions

Before a non-trivial design or structural change, verify against the governing docs
(precedence table above) — consult, don't guess:

1. **Requirements / data format** — does it touch a WMKO requirement or the L2/L4 data
   products? Check `WMKO_REQUIREMENTS.md` / `EPRV_DATA_STANDARD.md` first.
2. **Principles** — determinism, no hidden state, fail-loud, explicit calibration,
   simplicity (charter §9 Guardrails, §10 Core Design Principles; calibration §5).
3. **Structure** — matches the data-model / alias / layering / QC contracts
   (architecture reference).
4. **Change gate** — preserves deterministic behavior, runs on the truth dataset, and
   documents RV-metric impact (charter §6); measure against §3 Definition of Success.

The governing docs are the single source of truth — read them, don't restate them here
(past inline copies drifted). Intent & principles: [`KPF_VNEXT_CHARTER.md`](docs/dev/KPF_VNEXT_CHARTER.md). Structure (data model, aliases, layering, masters,
QC/checkpoint/quicklook, tests): [`KPF_VNEXT_ARCHITECTURE.md`](docs/dev/KPF_VNEXT_ARCHITECTURE.md). Code style: [`KPF_VNEXT_STYLE_GUIDE.md`](docs/dev/KPF_VNEXT_STYLE_GUIDE.md).

Do not modify CLAUDE.md without explicit instruction to do so from a human developer.
