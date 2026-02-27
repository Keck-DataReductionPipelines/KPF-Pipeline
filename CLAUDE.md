# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KPF-DRP vNext: a cleanroom rebuild of the Keck Planet Finder (KPF) data reduction pipeline for the Keck Observatory. This is not a cosmetic refactor — it is a controlled reset aimed at restoring scientific confidence, deterministic behavior, and development agility.

**The scientific priority is intermediate and long-term radial velocity (RV) stability.** Everything in this rebuild should serve: improving wavelength calibration robustness, improving drift modeling clarity, eliminating outliers, making calibration failure modes explicit, or enabling systematic regression testing. If a feature does not support RV stability or reproducibility, it is deprioritized.

### Path 3: Simple Astronomy-Style Pipeline

We are building explicit Python modules with file-in/file-out execution. No workflow framework, no service infrastructure, no database dependency, no orchestration engine, no premature architecture. The goal is clarity, reproducibility, and stability — not elegance. Path 2 (modern orchestration, containerization) is a future step once calibration logic stabilizes and RV performance is understood.

### Calibration Philosophy

The previous pipeline implicitly centered LFC as the foundational wavelength solution (WLS). In this rebuild: calibration paths (LFC, HCL, etalon) are separate, independently testable modules. No calibration source is trusted blindly. QC metrics must accompany every calibration product.

### Guardrails

Avoid: rebuilding a hidden framework unintentionally, adding implicit global state, embedding database dependencies in science code, introducing silent retries, overengineering abstractions.

## Development Environment

- **Python 3.14.3** (pinned exactly)
- **Conda env**: `kpfpipe` — set up via `conda env create -f KPF-Pipeline/environment.yml`
- **Install package**: `pip install -e KPF-Pipeline/` (editable install)
- **Key dependency**: `rv-data-standard` (RVData) from `git+https://github.com/EPRV-RCN/RVData.git@develop`

## Commands

```bash
# Run all tests (must run from KPF-Pipeline/ due to git receipt system requirement)
cd KPF-Pipeline && python -m pytest tests/ -v

# Run a single test class or test
python -m pytest tests/test_data_models.py::TestKPF2Aliases -v
python -m pytest tests/test_data_models.py::TestKPF2Aliases::test_chip_prefix_access -v

# Formatting and linting
black kpfpipe/ tests/
isort kpfpipe/ tests/
flake8 kpfpipe/ tests/
mypy kpfpipe/
```

## Architecture

### Data Model Hierarchy

Data products follow the EPRV RV Data Standard (rvdata) with KPF-specific extensions:

```
RVDataModel (rvdata)
├── KPFDataModel (base.py)         — KPF base: obs_id, filename conventions
│   ├── KPF0 (level0.py)          — Raw CCD data (L0)
│   └── KPF1 (level1.py)          — Assembled 2D frames (L1)
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
L0 (raw CCD) → ImageAssembly → ImageProcessing → L1 (assembled 2D)
L1 → SpectralExtraction → WavelengthCalibration → BarycentricCorrection → L1 (science-ready)
L1.to_kpf2() → KPF2 (extracted spectra, EPRV-compliant)
```

### Configuration

Extension definitions, trace mappings, and aliases are CSV-driven (`data_models/config/`). Detector parameters (CCD dimensions, order counts) live in `data_models/config/detector.toml` and are exposed via `kpfpipe.constants`.

### Masters Pipeline

`kpfpipe/modules/masters/` — stacks multiple observations to create bias, dark, flat, and wavelength solution (WLS) calibration products. Uses sigma-clipped statistics with streaming Welford's algorithm for large stacks.

### RVDataModel Base Class

The rvdata `RVDataModel` provides `extensions`, `headers`, `data` (all OrderedDicts), plus `create_extension()`, `set_data()`, `set_header()`, `from_fits()`, `to_fits()`, and a receipt system. The base `set_data()`/`set_header()` use `.keys()` checks that bypass `__contains__` overrides, so KPF2/KPF4 override these methods with a `hasattr` guard to resolve aliases during init before the dicts are upgraded.

## Design Principles

From the project charter — these guide all implementation decisions:

1. No hidden state
2. No implicit calibration assumptions
3. Deterministic stacking and wavelength solution
4. Fail loudly
5. Log everything relevant to calibration and RV derivation
6. Write QC metrics alongside products
7. Prefer clarity over cleverness
8. Implement everything in the simplest possible way.

Every major change must: preserve deterministic behavior, run on the truth dataset, and document impact on RV metrics.

- Keep this file updated as you make mistakes, learn lessons, and learn more efficient ways of coding or answering prompts.
- Use this file as your long-term memory.

## Success Criteria

The pipeline is successful when: it runs deterministically, can reprocess a frozen truth dataset without random failures, supports an independent HCL-based WLS path, calibration strategies are explicit and comparable, RV performance on standard stars is stable and measurable, and silent failures are eliminated.
