# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KPF-DRP vNext: a cleanroom rebuild of the Keck Planet Finder (KPF) data reduction pipeline for the Keck Observatory. The scientific priority is intermediate and long-term radial velocity (RV) stability.

**Four authoritative references govern this project, in strict precedence — when they conflict, the higher one wins:**

**1. The WMKO technical requirements — [`WMKO_REQUIREMENTS.md`](WMKO_REQUIREMENTS.md) (repo root; faithful mirror of `WMKO_REQUIREMENTS.pdf`) — are the highest authority: the W. M. Keck Observatory's binding technical requirements for the DRP (development, installation/build, runtime, archive). They outrank every reference below. The pipeline is still in active development, so MOST of these requirements are not yet met — this is expected. Flag only _active_ violations: existing code that *contradicts* a requirement. Do NOT flag _passive_ violations: a requirement unmet simply because the relevant code/feature does not exist yet. A missing capability is not a violation; code that does the wrong thing is.**

**2. The EPRV data standard — [`EPRV_DATA_STANDARD.md`](EPRV_DATA_STANDARD.md) (repo root) — is the source of truth for KPF's data products (L2/L4): FITS structure, extension and header-keyword names, units, and reference frames (vacuum wavelengths, BJD_TDB, barycentric frame). KPF L2/L4 are EPRV-compliant by contract, so the standard takes priority on anything touching data format. It mirrors the KPF-relevant portions of <https://eprv-data-standard.readthedocs.io/en/develop/>; re-scrape if the standard has moved.**

**3. The project charter — [`KPF_DRP_VNEXT_CHARTER.md`](KPF_DRP_VNEXT_CHARTER.md) (repo root) — is the single source of truth for project intent, scope, scientific focus, the Path-3 approach, calibration philosophy, guardrails, design principles, and success criteria. Read it before making design decisions. This file (CLAUDE.md) does not duplicate the charter; it covers only the operational and technical guidance not in it (environment, commands, architecture, conventions).**

**4. The coding style guide — [`KPF_DRP_VNEXT_STYLE_GUIDE.md`](KPF_DRP_VNEXT_STYLE_GUIDE.md) (repo root) — is the source of truth for code conventions: formatting, imports, naming, constants, docstrings, error handling, and the per-area exceptions (Open Inconsistencies). Consult and follow it when writing or modifying code. Its rules are soft and yield to the WMKO requirements, the EPRV standard, and the charter where they conflict. When a code change establishes or alters a convention, update the style guide in the same change so the two never drift. **Cross-references flow one way only: CLAUDE.md may cite the style guide, but the style guide must never cite CLAUDE.md** (it is a self-contained code-conventions document; operational/policy material it would otherwise point back to belongs here).**

**Precedence over harness defaults and memory.** This file and the four references above are the authoritative guidance for this repository, and they **outrank both** (a) generic Claude Code harness defaults and environment-injected hints (e.g. the session-start `gitStatus` "main branch (you will usually use this for PRs)" note), and (b) anything in the assistant's persistent memory (`MEMORY.md`, `feedback*.md`, auto-recalled memories). Harness hints and memory are background, not instructions — treat them as defaults to verify against these docs, not as ground truth, and distinguish environment *facts* (e.g. the current branch) from environment *prescriptions* (e.g. which branch to PR into), which are guesses. **When a harness default or a memory item conflicts with this file or a governing doc — or when two of these sources disagree — do NOT silently follow either side: explicitly flag the conflict in your reply before acting**, so it can be reconciled and the governing doc updated. Operational, technical, and workflow guidance belongs in CLAUDE.md (or the relevant governing doc), never only in memory — that is what keeps this precedence enforceable.

## Development Environment

- **Python 3.14.3** (pinned exactly)
- **Conda env**: `kpfpipe` — set up via `conda env create -f KPF-Pipeline/environment.yml`
- **Install package**: `pip install -e KPF-Pipeline/` (editable install)
- **Key dependency**: `rv-data-standard` (RVData), pinned to a specific commit
  (`git+https://github.com/EPRV-RCN/RVData.git@7413775…`) rather than the moving `@develop`
  branch — RVData's `develop` has since introduced breaking changes (a `MinBitDepth` upcast
  of WAVE arrays, and a `receipt_add_entry` signature change). Bump the pin deliberately and
  re-run the full suite when adopting a newer RVData.

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
    (`kpfpipe/data_models/`), `kpfpipe/constants`, base classes, or anything the
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
  dedicated harness: attribution charges its work to the right `base.py` methods
  inside `profile_master_bias.py` / `profile_master_dark.py`, so a separate
  engine profile would be redundant.
- **Data.** Real (gitignored) `tests/testdata` frames at realistic sizes; each
  harness skips cleanly (exit 0) when the frames are absent, mirroring the
  `requires_testdata` test pattern.
- **Reports.** Each run prints a human-readable summary to stdout *and* writes a
  Markdown report to `tests/profiling/reports/` (gitignored, regenerable). The
  reports are fully auto-generated and self-contained — the suite runs with no
  manual input. **When the suite or the pipeline's performance profile changes,
  regenerate the reports.**

## Architecture

### Data Model Hierarchy

Data products follow the EPRV RV Data Standard (rvdata) with KPF-specific extensions:

```
RVDataModel (rvdata)
└── KPFDataModel (base.py)         — shared KPF behavior (see below)
    ├── KPF0 (level0.py)                       — Raw CCD data (L0)
    ├── KPF1 (level1.py)                       — Assembled FFI (L1)
    ├── KPF2 (KPFDataModel, RV2) (level2.py)   — Extracted spectra (L2) with aliases
    └── KPF4 (KPFDataModel, RV4) (level4.py)   — RVs and CCFs (L4) with aliases
```

**All four models inherit `KPFDataModel`.** L0/L1 do so directly; L2/L4 via multiple inheritance alongside rvdata's `RV2`/`RV4` — `KPFDataModel` is listed **first** so its overrides win, while `RV2`/`RV4`'s `_read`/`from_fits`/level-specific `_create_hdul` stay reachable through `super()`. `KPFDataModel` is the single home for shared behavior: `obs_id`, `as_fits_header`, `create_extension`, alias-aware `set_data`/`set_header` (`hasattr`-guarded, inert for L0/L1), `receipt_add_entry`/`_update_drpstatus`, and `_create_hdul`/`_restore_primary_comments`. **`check_filename_convention` is the exception** — every concrete model declares it *explicitly* (even bare pass-throughs), and `KPFDataModel`'s version **raises `NotImplementedError`** (the base is abstract — only ever inherited). The conventions: `KP.*` (KPF0), `kpf_L1_*` (KPF1), the EPRV `SL#` check delegated to rvdata via `RV2`/`RV4` (KPF2/KPF4), and the DRP-RUN-05 master name `{KOAID}_master_{type}_L{N}.fits` on `KPFMasterModel` (which precedes KPF1/2/4 in the masters MRO, so it wins). L2/L4 add KPF-friendly extension aliases via `AliasedOrderedDict`.

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

### Header standardization (EPRV PRIMARY)

The WMKO-native → EPRV-standard PRIMARY conversion lives in **exactly one place**:
`KPF0.to_kpf1()`, which calls the conversion methods **`KPF0.wmko_to_eprv()`** and
**`KPF0.build_instrument_header()`** (L0 owns the WMKO→EPRV mapping). The keyword registry lives in
**`kpfpipe/data_models/keyword_registry.py`** as the single `KeywordRegistry` class (one module
singleton `keyword_registry`), organized by its three use-cases: **(1) mapping** — the rvdata
`header_map.csv` (`.header_map`); **(2) validation** — `.allowed`/`.required` per extension +
`.structural`; **(3) routing** — `.routing` (`keyword → (extension, comment)`). All of (2)/(3) are
derived in `__init__` from the **single source-of-truth table `.table`** — one DataFrame unioning our
`L{0,1,2,4}-headers.csv` registries with the EPRV keyword defs (rvdata's `LEVEL2/4_PRIMARY_KEYWORDS` +
per-extension CSVs), columns `Keyword, Description, Extension, DataType, PopulatedBy, Required, Level`
(`.registered` is the keyword allowlist). **`keyword_registry.py` is imported only by
`data_models/base.py`**: `KPFDataModel` surfaces the singleton as the **class attribute
`keyword_registry`** and re-exports it (so `level2/4.py` call `keyword_registry.register_rvdata_extension`
from `base`) — consumers handed a `kpf_obj` (the checkpoints validator, level0's WMKO→EPRV mapping,
tests) read the lookups off `kpf.keyword_registry.{table,routing,allowed,required,structural,registered,header_map}`
rather than importing the registry.
Consequences every contributor must respect:

- **L1 PRIMARY holds EPRV-registered keywords only.** From L1 onward, PRIMARY holds EPRV keyword
  *names* (incl. `DRPTAG`, `DATALVL`) plus FITS-standard structural/bookkeeping cards — **no
  KPF-registered keywords** (the DRP provenance cards live on RECEIPT; see below) and never raw
  WMKO natives.
- **`INSTRUMENT_HEADER` is an immutable, _pure_ verbatim copy of the raw L0 PRIMARY** (values
  **and** comments — `build_instrument_header()` returns a `fits.Header` copy). It is the raw
  instrument cards only — the pipeline stamps **nothing** onto it (DRP provenance goes to RECEIPT,
  not the L0 PRIMARY, so it never appears here). Written once in `to_kpf1`; **nothing else ever
  writes to it**. Code that needs a raw instrument keyword (e.g. `ELAPSED`, `MJD-OBS`, `DATE-OBS`,
  `GAIAID`, `SCI-OBJ`, `TARGTEFF`) reads it from `INSTRUMENT_HEADER`, not PRIMARY.
- **DRP-RUN provenance is stamped onto the L0 RECEIPT at read (`KPF0.from_fits`), not at `to_kpf1`.**
  `KPF0._stamp_provenance` (the single population site; their `PopulatedBy` in `L0-headers.csv` is
  `KPF0.from_fits`, `Extension` is `RECEIPT`) writes `DRPVERNO` (DRP-RUN-11), `DRPSTATU` (DRP-RUN-20),
  and `PROGID`/`KOAID` (DRP-RUN-19) via `set_keyword` (so KPF0 carries a RECEIPT extension —
  `L0-extensions.csv` marks it Required). `PROGID`/`KOAID` are read from the WMKO-native PRIMARY; an
  absent value is defaulted to `UNKNOWN` **with a warning**. They then ride the RECEIPT-header
  forward (`to_kpf1`→`to_kpf2`→`to_kpf4`) downstream — the raw PRIMARY and INSTRUMENT_HEADER are left
  untouched. `DRPSTATU` is advanced per module by `_update_drpstatus`, also to RECEIPT.
- **KPF-pipeline keywords are written via `KPFDataModel.set_keyword(key, value)`, NOT directly to a
  header.** `set_keyword` looks the keyword up in the registry and writes it to the **extension named
  in the registry's `Extension` column**, with the registry `Description` as the FITS comment — so a
  keyword always lands on the same extension with the same comment, and writers never name an
  extension or comment. It **fails loudly**: `KeyError` if the keyword is unregistered, `ValueError`
  if its home extension does not exist on the object. Every such keyword must be registered in the
  per-level registry `data_models/config/L{0,1,2,4}-headers.csv` (explicit ≤8-char FITS keyword — no
  wildcards; families enumerated per member, e.g. `RNGREEN1`-`RNGREEN4`). The current homes:
  - **PRIMARY** — EPRV-registered keywords only: `DRPTAG`, `DATALVL`, and the EPRV RV cards (no
    KPF-registered keyword routes to PRIMARY).
  - **QUALITY_CONTROL** (a KPF-only `BinTableHDU`, created on KPF0/1/2/4) — QC booleans + `ISGOOD`,
    read-noise (`RN*`), calibration **ages** (`BIASAGE`/`DARKAGE`/`FLATAGE`/`WLSAGE`), and DiagL2
    metrics (`NAN*`/`ZEROFRAC`/`*SNR*`/`*FR*`).
  - **RECEIPT** — DRP provenance (`PROGID`/`KOAID`/`DRPVERNO`/`DRPSTATU`, stamped at read), `ORIGID`,
    applied flags (`OSCANSUB`/`BIASSUB`/`DARKSUB`/`FLATDIV`), calibration **paths**
    (`BIASFILE`/`DARKFILE`/`FLATFILE`/`WLSFILE`), `ASTRSRC`.
  - **BJD_TDB / BARYCORR_KMS / BARYCORR_Z** (EPRV L2 extensions) — the per-CCD barycentric summaries.
  - **RV1–RV5** (EPRV L4 tables) — the per-orderlet legacy RVs (`CCD<n>RV<sfx>`, sfx S→RV1, 1→RV2,
    2→RV3, 3→RV4, C→RV5) plus the SCI-combined RV summaries on **RV3**
    (`CCD1RV`/`CCD2RV`/`CCD1ERV`/`CCD2ERV`, `CCFRV`, `CCFERV`). Masters keep `MASTYPE` on their own
    PRIMARY (out of EPRV scope).
- **QUALITY_CONTROL + RECEIPT *headers* propagate L0→L1→L2→L4** card-by-card in
  `to_kpf1`/`to_kpf2`/`to_kpf4` (`set_header` with a comment-preserving `fits.Header` copy),
  alongside the receipt *table* copy. All three `to_kpfN` methods share one invariant — forward both
  governed headers if present — so QUALITY_CONTROL is an **append-only history of every diagnostic /
  QC flag**, exactly as RECEIPT is the append-only history of processing steps. The **only** QC
  keyword that changes level-to-level is **`ISGOOD`**, the running aggregate: each level's `QC.run`
  recomputes it as the AND over *every* QC flag then on QUALITY_CONTROL (propagated + just-written,
  via `keyword_registry.qc_flag_keywords`, excluding `ISGOOD` itself), so it reflects all QC
  accumulated so far. The `qc_flags` checkpoint, by contrast, warns/raises only on the **current
  level's own** flags (`keyword_registry.qc_flag_keywords_by_level[LEVEL]`) — a propagated
  lower-level `0` was already surfaced at its own level, so it is not re-warned downstream.
  QUALITY_CONTROL is a KPF-custom extension unknown to rvdata's definition-driven L2/L4 reader, so
  `keyword_registry.register_rvdata_extension` (a method on `KeywordRegistry`, re-exposed via `base`,
  called from `level2.py`/`level4.py`) registers it into rvdata's in-memory `LEVEL{2,4}_EXTENSIONS` so
  a product written with it reads back.
- **Header validation lives in the `quality_control/checkpoints` layer
  (`Checkpoint.unregistered_keywords`), NOT in QC or `to_kpf2`/`to_kpf4`.** QC writes only 0/1
  flags; the Checkpoint stage runs after it (science → Diagnostics → QC → Checkpoints) and reads
  the flags + headers to warn or raise. For each registry-governed extension present (PRIMARY,
  QUALITY_CONTROL, RECEIPT, the barycentric extensions, RV#/CCF#), every card must be a registered
  keyword for that extension (`keyword_registry.allowed`) or a structural card
  (`keyword_registry.is_structural` — the **single** structural definition: exact FITS bookkeeping
  cards in `_STRUCTURAL` plus the WCS/bintable prefix families in `_STRUCTURAL_PREFIXES`; the
  checkpoint delegates to it and keeps no list of its own), else it **raises**.
  PRIMARY is validated only where it is EPRV-standard (L1/L2/L4); the raw WMKO L0 PRIMARY is
  skipped. The "required keywords present" concern is now a **QC flag** instead: `KWRDPRL{1,2,4}`
  is 1 iff every registry-required PRIMARY keyword for the level is present (EPRV `Required` at
  `Level ≤ product level`; PRIMARY now holds EPRV-registered keywords only, so there is no KPF-routed
  set to union in); `KWRDPRL0` stays hardcoded (raw L0 PRIMARY is not registry-governed). The `qc_flags` checkpoint then reads each
  0/1 flag and **raises** it if it is in the level's `RAISE_FLAGS` (data-present only), else
  **warns**. Checkpoints read all lookups off the validated `kpf_obj`
  (`self.kpf_obj.keyword_registry`), so `quality_control` imports nothing from `data_models`.
- **`wmko_to_eprv` emits only registered keywords.** It applies `header_map.csv` but **filters each
  `STANDARD` target against the registry** (via `self.keyword_registry.registered`), so rvdata's
  header_map entries that aren't EPRV keywords (`PARANG`, `PARANG2`) are dropped, never leaking onto
  PRIMARY (the raw value survives in `INSTRUMENT_HEADER`). `KeywordRegistry` warns once at construction
  listing any such header_map/registry inconsistency.
- `to_kpf1` also corrects values the installed `header_map.csv` gets wrong: `NUMORDER` (→67 from
  `DETECTOR`), `JD_UTC` (full JD from `MJD-OBS`, the canonical KPF exposure time — native
  `DATE-OBS` is date-only), and stamps the EPRV DRP version `DRPTAG` (its WMKO equivalent `DRPVERNO`
  is stamped at read; see above). Calibration masters are out of EPRV scope (their products keep
  their own layout).
- **Every extension header is an `astropy.io.fits.Header`.** `KPFDataModel.create_extension`
  (`data_models/base.py`) stores a new header as a `fits.Header`, not rvdata's default
  `OrderedDict`; `from_fits` already returns `fits.Header`. So **read with `header.get(key)` /
  `header[key]` and write with `header[key] = (value, comment)`** — native astropy, no
  value-vs-`(value, comment)` ambiguity and no header-parser helper. rvdata's base `_create_hdul`
  serializes PRIMARY by iterating `.items()`, which drops a `fits.Header`'s comments; the single
  `KPFDataModel._create_hdul` override (inherited by all four models) rebuilds the PRIMARY HDU via
  `self._restore_primary_comments(...)` so comments survive `to_fits`. Conversion lives in
  `KPF0.wmko_to_eprv`/`build_instrument_header` (`data_models/level0.py`); the unified
  `KeywordRegistry` table and its derived routing/validation lookups live in
  `data_models/keyword_registry.py` (the `keyword_registry` singleton), surfaced through `KPFDataModel`
  (`set_keyword` + the `keyword_registry` class attribute); header validation itself lives in
  `quality_control/checkpoints/base.py` (`Checkpoint.unregistered_keywords`).

### Configuration

Extension definitions, trace mappings, and aliases are CSV-driven (`data_models/config/`). Detector parameters (CCD dimensions, order counts) live in `data_models/config/detector.toml` and are exposed via `kpfpipe.constants`.

### Filename conventions

Science: `L0 = {obs_id}.fits` (KPF-native `KP.*`); `L1 = kpf_L1_{YYYYMMDD}T{HHmmss}.fits` — note **no EPRV "S"**, because the EPRV standard defines no L1 (its filename regex only accepts `SL2`/`SL3`/`SL4`); `L2/L4 = kpf_SL{N}_…` (EPRV-standard). Masters (WMKO DRP-RUN-05): `{KOAID-of-first-input}_master_{type}_L{N}.fits`.

Two authorities encode this rule and **must agree per level**: `build_filepath(obs_id, level, …)` (`utils/io.py`) is the pipeline's path builder (directory + filename, from an obs_id string) and is what recipes use to write; `<model>.generate_standard_filename()` builds only the basename from a populated object's headers and is the `to_fits(fn=None)` fallback. Like `check_filename_convention`, **every concrete model declares it explicitly** (KPF0/KPF1 build the KPF names; KPF2/KPF4 are bare pass-throughs to rvdata's EPRV builder via `RV2`/`RV4`; `KPFMasterModel` builds the master name), and `KPFDataModel`'s abstract version **raises `NotImplementedError`**. `TestFilenameConsistency` (in `tests/regression/test_masters_recipe.py`) enforces that this and `build_filepath` never drift.

### Masters Pipeline

`kpfpipe/modules/masters/` — stacks multiple observations to create bias, dark, flat, and wavelength solution (WLS) calibration products. Uses sigma-clipped statistics with a single-pass streaming accumulation (per-pixel counts and exposure time) for large stacks; the master image is the exposure-weighted rate `counts_sum / exptime_sum`.

### RVDataModel Base Class

The rvdata `RVDataModel` provides `extensions`, `headers`, `data` (top-level OrderedDicts), plus `create_extension()`, `set_data()`, `set_header()`, `from_fits()`, `to_fits()`, and a receipt system. The base `set_data()`/`set_header()` use `.keys()` checks that bypass `__contains__` overrides, so KPF2/KPF4 override these methods with a `hasattr` guard to resolve aliases during init before the dicts are upgraded. The base `create_extension()` initializes each extension *header* as an `OrderedDict`; the KPF models override it so every header is a `fits.Header` instead (see *Header standardization*).

### Diagnostics, QC, Checkpoints, and Quicklook

Four read-only layers, consolidated under `kpfpipe/quality_control/`, consume data products. None of them mutate the scientific arrays — they only read data and write header keywords via `set_keyword` (routed to QUALITY_CONTROL — see *Header standardization*) (and, in Quicklook's case, to PNG files). Per-level files follow the `levelN.py` naming used by `data_models/`. The first three run in a strict order — **Diagnostics → QC → Checkpoints** — each consuming what the prior wrote. The recipe drives all three through a **single `CheckpointL{n}(obj).run()` call**: `Checkpoint.run()` folds in the paired Diagnostics and QC classes first (named on the subclass as the `DIAGNOSTICS`/`QC` class attributes, e.g. `CheckpointL1.DIAGNOSTICS = DiagL1`), then runs the checkpoint methods — so callers no longer invoke `DiagL{n}`/`QCL{n}` directly. The folded `QC.run()` result dict is captured on `Checkpoint.qc_results` for reporting (e.g. `scripts/qc.py`). A level with no paired class skips that stage; `CheckpointL0` is wired but the recipe leaves L0 unrun (running QCL0 pre-assembly would leak L0 flags onto L1/L2 via header propagation).

- **Diagnostics** (`kpfpipe/quality_control/diagnostics/`) — computes scalar/array metrics from finished data products and writes them via `set_keyword` (DiagL2 metrics land on QUALITY_CONTROL). Per-level classes (`DiagL0`/`DiagL1`/`DiagL2`) mirror the QC structure. Examples: per-fiber NaN counts in extracted spectra, zero-flux fraction.
- **QC** (`kpfpipe/quality_control/qc_flags/`) — reads metrics (mostly from headers populated by Diagnostics or pipeline modules) and applies pass/fail thresholds. Writes **only** 0/1 keywords (via `set_keyword`, routed to QUALITY_CONTROL) plus the `ISGOOD` aggregate. `ISGOOD` is the **running** aggregate — the AND over every QC flag accumulated on QUALITY_CONTROL so far (this level's checks *plus* those propagated from lower levels), not just this level's checks. No validation or raising — that is the Checkpoints layer's job.
- **Checkpoints** (`kpfpipe/quality_control/checkpoints/`) — reads the 0/1 QC flags and the product headers and **emits warnings or raises errors** (never writes). Two inherited base checkpoints: `unregistered_keywords` (structural header validation — see *Header standardization*) and `qc_flags` (raises a failed flag named in the per-level `RAISE_FLAGS`, warns the rest) — scoped to the **current level's own** flags (`keyword_registry.qc_flag_keywords_by_level[LEVEL]`), so a propagated lower-level `0` is not re-warned. `CheckpointL0`/`L1`/`L2` set `LEVEL` + `RAISE_FLAGS`.
- **Quicklook** (`kpfpipe/quality_control/quicklook/`) — reads products and renders matplotlib plots. Pulls any annotation values from existing headers.

This is unlike v2.12, which had one big `DiagnosticsFramework` primitive with a conditional dispatch tree over many functions and shared backend state with `AnalyzeL0/2D/L1/L2` classes. v3 uses per-level classes with method-attribute registration (`_diag_name` / `_qc_key` / `_checkpoint_name`) and no shared state.

**Where metrics live.** Metrics that depend on intermediate processing state (e.g. read noise from raw overscan) stay in the pipeline module that produces them — they cannot be recomputed from the finished product. Metrics that can be computed from the finished product alone live in Diagnostics — including the master calibration **ages** (`BIASAGE`/`DARKAGE`/`FLATAGE`/`WLSAGE`), which `DiagL1` recomputes from the master paths `CalibrationAssociation` wrote to RECEIPT (`*FILE`) plus the `INSTRUMENT_HEADER` `DATE-OBS`; the association module writes only the paths.

**Detector geometry.** Helpers like `count_amplifiers`, `orient_channels`, and `_RN_KEYS` are owned by `ImageAssembly`. Other consumers (Quicklook, future Diagnostics) import them rather than duplicating the logic.

## Design Principles & Success Criteria

These live in the charter and are NOT restated here (the two copies drifted in the past — keep one source). See [`KPF_DRP_VNEXT_CHARTER.md`](KPF_DRP_VNEXT_CHARTER.md): §10 Core Design Principles, §9 Guardrails, §5 Calibration Philosophy, §3 Definition of Success, §6 (every major change must preserve deterministic behavior, run on the truth dataset, and document impact on RV metrics). Consult the charter before design decisions.

- Keep this file (CLAUDE.md) updated with operational lessons, conventions, and more efficient workflows learned while coding.
- Use CLAUDE.md as long-term memory for technical/operational guidance; use the charter for project intent and principles.
