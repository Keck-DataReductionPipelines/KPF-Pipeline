# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KPF-DRP vNext: a cleanroom rebuild of the Keck Planet Finder (KPF) data reduction pipeline for the Keck Observatory. The scientific priority is intermediate and long-term radial velocity (RV) stability.

**The project charter — [`KPF_DRP_VNEXT_CONTEXT.md`](KPF_DRP_VNEXT_CONTEXT.md) (repo root) — is the single source of truth for project intent, scope, scientific focus, the Path-3 approach, calibration philosophy, guardrails, design principles, and success criteria. Read it before making design decisions. This file (CLAUDE.md) does not duplicate the charter; it covers only the operational and technical guidance not in it (environment, commands, architecture, conventions).**

**The coding style guide — [`KPF_DRP_VNEXT_STYLE_GUIDE.md`](KPF_DRP_VNEXT_STYLE_GUIDE.md) (repo root) — is the source of truth for code conventions: formatting, imports, naming, constants, docstrings, error handling, and the per-area exceptions (Open Inconsistencies). Consult and follow it when writing or modifying code. Its rules are soft and yield to the charter where they conflict. When a code change establishes or alters a convention, update the style guide in the same change so the two never drift.**

## Development Environment

- **Python 3.14.3** (pinned exactly)
- **Conda env**: `kpfpipe` — set up via `conda env create -f KPF-Pipeline/environment.yml`
- **Install package**: `pip install -e KPF-Pipeline/` (editable install)
- **Key dependency**: `rv-data-standard` (RVData), pinned to a specific commit
  (`git+https://github.com/EPRV-RCN/RVData.git@7413775…`) rather than the moving `@develop`
  branch — RVData's `develop` has since introduced breaking changes (a `MinBitDepth` upcast
  of WAVE arrays, and a `receipt_add_entry` signature change). Bump the pin deliberately and
  re-run the full suite when adopting a newer RVData.

## Commands

```bash
# Run all tests (must run from KPF-Pipeline/ due to git receipt system requirement)
cd KPF-Pipeline && python -m pytest tests/ -v

# Run a single test class or test
python -m pytest tests/test_data_models.py::TestKPF2Aliases -v
python -m pytest tests/test_data_models.py::TestKPF2Aliases::test_chip_prefix_access -v

# Formatting and linting (Ruff; config in pyproject.toml [tool.ruff])
ruff format kpfpipe/ tests/ recipes/      # format (black-compatible)
ruff check --fix kpfpipe/ tests/ recipes/ # lint + auto-fix

# Pre-commit hook (enforces ruff format + lint on commit)
pre-commit install          # one-time, after creating the env
pre-commit run --all-files  # run all hooks across the repo
```

## Architecture

### Data Model Hierarchy

Data products follow the EPRV RV Data Standard (rvdata) with KPF-specific extensions:

```
RVDataModel (rvdata)
├── KPFDataModel (base.py)         — KPF base: obs_id, filename conventions
│   ├── KPF0 (level0.py)          — Raw CCD data (L0)
│   └── KPF1 (level1.py)          — Assembled FFI (L1)
├── RV2 (rvdata)
│   └── KPF2 (level2.py)          — Extracted spectra (L2) with aliases
└── RV4 (rvdata)
    └── KPF4 (level4.py)          — RVs and CCFs (L4) with aliases
```

L0 and L1 subclass `KPFDataModel` (which wraps `RVDataModel`). L2 and L4 subclass rvdata's `RV2`/`RV4` directly and add KPF-friendly extension aliases via `AliasedOrderedDict`.

### Extension Alias System

`AliasedOrderedDict` (aliased_dict.py) transparently maps KPF names to EPRV-standard names. Designed to be generic enough to upstream into rvdata.

KPF2 aliases (driven by CSV configs in `data_models/config/`):
- **Fiber aliases**: `SCI2_FLUX` → `TRACE3_FLUX`, `CAL_WAVE` → `TRACE1_WAVE`
- **Simple aliases**: `CA_HK` → `ANCILLARY_SPECTRUM`, `EXPMETER_SCI` → `EXPMETER`
- **Chip-prefix access**: `GREEN_SCI2_FLUX` returns `TRACE3_FLUX[:35]` as a numpy view (sliced at `NORDER_GREEN`). Handled by `_KPF2DataDict`, a subclass of `AliasedOrderedDict`.

Traces store 67 orders concatenated (35 green + 32 red). Chip-prefix keys are computed views, not separate storage.

### Data Flow

```
L0 (raw CCD) → ImageAssembly → ImageProcessing → L1 (assembled FFI)
L1 → SpectralExtraction → WavelengthCalibration → BarycentricCorrection → L1 (science-ready)
L1.to_kpf2() → KPF2 (extracted spectra, EPRV-compliant)
```

### Configuration

Extension definitions, trace mappings, and aliases are CSV-driven (`data_models/config/`). Detector parameters (CCD dimensions, order counts) live in `data_models/config/detector.toml` and are exposed via `kpfpipe.constants`.

### Masters Pipeline

`kpfpipe/modules/masters/` — stacks multiple observations to create bias, dark, flat, and wavelength solution (WLS) calibration products. Uses sigma-clipped statistics with streaming Welford's algorithm for large stacks.

### RVDataModel Base Class

The rvdata `RVDataModel` provides `extensions`, `headers`, `data` (all OrderedDicts), plus `create_extension()`, `set_data()`, `set_header()`, `from_fits()`, `to_fits()`, and a receipt system. The base `set_data()`/`set_header()` use `.keys()` checks that bypass `__contains__` overrides, so KPF2/KPF4 override these methods with a `hasattr` guard to resolve aliases during init before the dicts are upgraded.

### Diagnostics, QC, and Quicklook

Three read-only layers, consolidated under `kpfpipe/quality_control/`, consume data products. None of them mutate the scientific arrays — they only read data and write to PRIMARY headers (and, in Quicklook's case, to PNG files). Per-level files follow the `levelN.py` naming used by `data_models/`.

- **Diagnostics** (`kpfpipe/quality_control/diagnostics/`) — computes scalar/array metrics from finished data products and writes them to PRIMARY headers. Per-level classes (`DiagL0`/`DiagL1`/`DiagL2`) mirror the QC structure. Examples: per-fiber NaN counts in extracted spectra, zero-flux fraction.
- **QC** (`kpfpipe/quality_control/qc_booleans/`) — reads metrics (mostly from headers populated by Diagnostics or pipeline modules) and applies pass/fail thresholds. Writes 0/1 keywords plus `ISGOOD` aggregate.
- **Quicklook** (`kpfpipe/quality_control/quicklook/`) — reads products and renders matplotlib plots. Pulls any annotation values from existing headers.

This is unlike v2.12, which had one big `DiagnosticsFramework` primitive with a conditional dispatch tree over many functions and shared backend state with `AnalyzeL0/2D/L1/L2` classes. v3 uses per-level classes with method-attribute registration (`_diag_name` / `_qc_key`) and no shared state.

**Where metrics live.** Metrics that depend on intermediate processing state (read noise from raw overscan, master ages from header lookups during association) stay in the pipeline module that produces them — they cannot be recomputed from the finished product. Metrics that can be computed from the finished product alone live in Diagnostics.

**Detector geometry.** Helpers like `count_amplifiers`, `orient_channels`, and `_RN_KEYS` are owned by `ImageAssembly`. Other consumers (Quicklook, future Diagnostics) import them rather than duplicating the logic.

## Design Principles & Success Criteria

These live in the charter and are NOT restated here (the two copies drifted in the past — keep one source). See [`KPF_DRP_VNEXT_CONTEXT.md`](KPF_DRP_VNEXT_CONTEXT.md): §10 Core Design Principles, §9 Guardrails, §5 Calibration Philosophy, §3 Definition of Success, §6 (every major change must preserve deterministic behavior, run on the truth dataset, and document impact on RV metrics). Consult the charter before design decisions.

- Keep this file (CLAUDE.md) updated with operational lessons, conventions, and more efficient workflows learned while coding.
- Use CLAUDE.md as long-term memory for technical/operational guidance; use the charter for project intent and principles.
