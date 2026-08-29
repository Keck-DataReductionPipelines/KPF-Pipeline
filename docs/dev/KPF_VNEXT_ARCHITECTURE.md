# KPF-DRP vNext: Pipeline Architecture

This document describes the repo structure and KPF vNext pipeline **architecture**;
it describes *how the pipeline is structured* — not project intent (the charter) or
code conventions (the style guide). Consult this document before making structural
or cross-cutting changes.

Contents:

* Repo Structure
* Data Flow
  * Science Pipeline
  * Masters Pipeline
* Data Models
  * EPRV Data Standard
  * Extension alias system
  * Header standardization
  * Keyword registry
  * Masters data models
* Processing layers
  * Modules
  * Recipes & configs
  * Scripts
  * Command line interface (CLI)
* Quality control
  * Diagnostics
  * QC flags
  * Checkpoints
  * Quicklook plots
* Logging
* File handling
  * Data directory structure
  * FileHandler class
  * io.py path builders
* Tests
  * Regression
  * Profiling


**Authority precedence.**
When requirements or design principles conflict, the order of governing document precedence is:

1. WMKO technical requirements ([`WMKO_REQUIREMENTS.md`](WMKO_REQUIREMENTS.md))
2. EPRV data standard ([`EPRV_DATA_STANDARD.md`](EPRV_DATA_STANDARD.md))
3. KPF vNext project charter ([`KPF_VNEXT_CHARTER.md`](KPF_VNEXT_CHARTER.md))
4. KPF vNext architecture reference ([`KPF_VNEXT_ARCHITECTURE.md`](KPF_VNEXT_ARCHITECTURE.md))
5. KPF vNext style guide ([`KPF_VNEXT_STYLE_GUIDE.md`](KPF_VNEXT_STYLE_GUIDE.md))

When any two conflict, the higher one wins.

---

## Repo Structure

The repository is organized into a small number of top-level packages, layered so each
imports only from those below it (see *Processing layers*):

```
kpfpipe/            scientist-facing building blocks (importable, no orchestration)
  data_models/      KPF0/1/2/4 + masters models, keyword registry, alias dicts, config CSVs
  modules/          processing algorithms (science + masters)
  quality_control/  quality control layers (diagnostics, qc_flags, checkpoints, quicklook)
  utils/            shared helpers (io, logger, config, stats, astro, kpf)
recipes/            compose modules into an end-to-end reduction (kpf_drp_{science,masters}.py)
configs/            default recipe parameters (kpf_drp_{science,masters}.toml)
scripts/            run recipes many times (processing/, plots/, quality_control/)
tools/              the `kpfpipe` CLI dispatcher (cli.py) and operator tools
reference/          static reference data (detector.toml, line lists, order traces, etc.)
tests/              regression/ (test suite) + profiling/ (performance harnesses)
docs/               Sphinx site (source/) and the governing docs (dev/)
```

A few top-level default quantities/constants are exposed and loaded by `kpfpipe/__init__.py`:
static detector parameters (e.g. pixel dimensions, amplifiers; `reference/detector.toml`),
CCD/trace identifiers (`kpfpipe.DEFAULTS`), and the repo root (`kpfpipe.REPO_ROOT`).

## Data Flow

The pipeline has two data flows: the **science** flow reduces a single observation from raw
CCD frames to RVs, and the **masters** flow stacks many calibration frames into nightly
master calibrations that the science flow consumes.

### Science Pipeline

```
L0 (raw CCD) → AstroQuery → ImageAssembly → L1 → ImageProcessing
L1 (assembled FFI) → SpectralExtraction → L2 → WavelengthCalibration → BarycentricCorrection
L2 (extracted spectra) → CrossCorrelation → L4 → RadialVelocity
L4 (CCFs/RVs)
```

The `kpf_drp_science` recipe drives this flow for one observation; the `science` CLI
orchestrator fans it out across many. Each `kpfpipe/modules/` algorithm processes a single
self-contained step of the data flow (see *Modules*).

### Masters Pipeline

`kpfpipe/modules/masters/` stacks multiple calibration observations into bias, dark, flat, and
wavelength-solution (WLS) products.

- **Stacking** is sigma-clipped and uses single-pass streaming accumulation (per-pixel counts +
  exposure time), so a large stack never holds every frame in memory at once; the master is the
  exposure-weighted rate (total counts ÷ total exposure).
- **The stage is I/O-bound**, so the shared engine (`base.py`) caches the first-read assembled
  frames for later passes to reuse instead of re-reading them.
- **Fan-out concurrency** is tuned separately from science (see *Scripts*); the resulting ML1/ML2
  products are described under *Masters data models*.

## Data Models

Data products follow the EPRV RV Data Standard (rvdata) with KPF-specific extensions. Extension definitions, trace mappings, and aliases are CSV-driven (`data_models/config/`).

### EPRV Data Standard

The class hierarchy:

```
RVDataModel (rvdata)
└── KPFDataModel (base.py)         — shared KPF behavior (see below)
    ├── KPF0 (level0.py)                       — Raw CCD data (L0)
    ├── KPF1 (level1.py)                       — Assembled FFI (L1)
    ├── KPF2 (level2.py)                       — Extracted spectra (L2) with aliases
    └── KPF4 (level4.py)                       — RVs and CCFs (L4) with aliases
```

**All four models inherit `KPFDataModel` directly.** Each declares its `level`, which names the
`config/L{n}-extensions.csv` manifest it builds from and the keyword profile it seeds PRIMARY with.

- **Shared behavior lives in `KPFDataModel`**: `obs_id`, `as_fits_header`, `create_extension`,
  alias-aware `set_data`/`set_header` (`hasattr`-guarded, inert for L0/L1),
  `receipt_add_entry`/`_update_drpstatus`, and `_create_hdul`/`_restore_primary_comments`.
- **`check_filename_convention` is the exception** — every concrete model declares it *explicitly*
  (even bare pass-throughs), and `KPFDataModel`'s version **raises `NotImplementedError`** (the base
  is abstract — only ever inherited). The conventions: `KP.*` (KPF0), `kpf_L1_*` (KPF1), the EPRV
  `SL#` check delegated to `RVDataModel.check_filename_convention` (KPF2/KPF4), and the DRP-RUN-05 master name
  `{KOAID}_master_{type}_L{N}.fits` on `KPFMasterModel` (which precedes KPF1/2/4 in the masters MRO,
  so it wins).
- **L2/L4 add KPF-friendly extension aliases** via `AliasedOrderedDict`.

`KPFDataModel` overrides `set_data`/`set_header` in `base.py` (not the level classes) with a `hasattr` guard so alias resolution runs during init — the rvdata base's `.keys()` checks would otherwise bypass the `__contains__` overrides — yet stays inert for non-aliased L0/L1. Its `create_extension` override makes every extension header a `fits.Header` rather than an `OrderedDict` (see *Header standardization*).

### Extension alias system

`AliasedOrderedDict` (aliased_dict.py) transparently maps KPF names to EPRV-standard names. Designed to be generic enough to upstream into rvdata.

KPF2 aliases (driven by CSV configs in `data_models/config/`):
- **Fiber aliases**: `SCI2_FLUX` → `TRACE3_FLUX`, `SKY_WAVE` → `TRACE1_WAVE`, `CAL_WAVE` → `TRACE5_WAVE`
- **Simple aliases**: `CA_HK` → `ANCILLARY_SPECTRUM`, `EXPMETER_SCI` → `EXPMETER`
- **Chip-prefix access**: `GREEN_SCI2_FLUX` returns `TRACE3_FLUX[:35]` as a numpy view (sliced at `NORDER_GREEN`). Handled by `_KPF2DataDict`, a subclass of `AliasedOrderedDict`.

Traces store 67 orders concatenated (35 green + 32 red). Chip-prefix keys are computed views, not separate storage.

### Header standardization

The WMKO-native → EPRV-standard PRIMARY conversion lives in **exactly one place** — the
`StandardizeDataFormat` module, which runs on the line after every raw-L0 load and also snapshots the
raw L0 PRIMARY verbatim into `INSTRUMENT_HEADER`.
The mapping, validation, and routing all derive from the keyword registry (see *Keyword registry*).
The architecture invariants:

- **PRIMARY holds EPRV-registered keywords only** from L0-after-standardization onward (EPRV keyword
  names + FITS structural cards — no KPF-registered keywords, no raw natives). `StandardizeDataFormat`
  seeds the whole registered PRIMARY skeleton for the level, then fills it from
  `EPRV-header-map.csv`; every card is present, blank where nothing supplied a value. (One
  keyword-homing exception is noted under *Keyword registry*.)
- **`INSTRUMENT_HEADER` is an immutable verbatim copy of the raw L0 PRIMARY** (values and comments),
  written once by `StandardizeDataFormat` and never again.
- **Read from PRIMARY, fall back to `INSTRUMENT_HEADER`** — at L0 too, now that standardization runs
  at load. The map carries only some natives to PRIMARY, mostly under renamed EPRV keys — so read a
  native from PRIMARY when it survives there under its own name (e.g. `DATE-OBS`, `OBJECT`), and from
  `INSTRUMENT_HEADER` when it never reaches PRIMARY or when a coherent block of related natives reads
  more clearly together (e.g. the raw `DATE-BEG`/`DATE-END` pair `barycentric_correction` extrapolates
  the exposure meter against; its *target astrometry*, by contrast, comes off the PRIMARY `C*#` cards,
  which `AstroQuery` fills from `CATALOG_RECORD`).
- **DRP provenance is stamped at read** onto RECEIPT (`KPF0.from_fits` → `_stamp_wmko_tracking`, not
  `to_kpf1`): `DRPVERNO`/`DRPSTATU`/`PROGID`/`KOAID`/`ORIGID`. It rides RECEIPT forward, with `DRPSTATU`
  advanced per module. `ORIGID` (the original L0 obs_id) is also how L1/L2/L4 recover `self.obs_id` on
  read, so every model carries `obs_id` on every construction path.
- **`QUALITY_CONTROL` + `RECEIPT` propagate L0→L1→L2→L4** card-by-card (`KPFDataModel._forward_headers`)
  as an **append-only history**.
- **`CATALOG_RECORD` (AstroQuery's resolved catalog rows) also passes through
  L0→L1→L2→L4**, and `AstroQuery` overlays its merged `kpf-drp` row onto the PRIMARY `C*#` cards.
- **Structural header validation lives in the checkpoints layer** (`Checkpoint.unregistered_keywords`),
  not in QC or `to_kpfN`: every card on a registry-governed extension must be a registered keyword or a
  structural card, else it raises.
- **Every extension header is an `astropy.io.fits.Header`** (not an `OrderedDict`;
  `KPFDataModel.create_extension` override), so keyword comments survive `to_fits`; the `_create_hdul`
  override only syncs the receipt table into RECEIPT.

### Keyword registry

The keyword registry (`kpfpipe/data_models/keyword_registry.py`) is a single `KeywordRegistry` class
with one module singleton `keyword_registry`, imported **only** by `data_models/base.py` and surfaced as
the `KPFDataModel.keyword_registry` class attribute. It derives its mapping/validation/routing lookups
from a **single source-of-truth table** unioning the
`config/{prefix}-{EXTENSION}-keywords.csv` registries, one file per extension per level.

**Each registered keyword has one home extension** (the registry `Extension` column) that `set_keyword`
routes to: **PRIMARY** (EPRV keywords), **QUALITY_CONTROL** (QC flags, read-noise,
calibration ages, DiagL2 metrics), **RECEIPT** (DRP provenance, applied flags, calibration paths),
the **barycentric** L2 extensions, and **RV1–RV5** (L4 per-orderlet `CCD{1,2}RV<sfx>`). The one exception to
*PRIMARY holds EPRV keywords only* is the L4 SCI-combined RV keywords `CCD{1,2}RV`/`CCD{1,2}ERV` —
KPF-registered yet homed on PRIMARY, since they are the pipeline's final RV measurements and belong
beside the EPRV `RV`/`RVERR`. Masters register their PRIMARY keywords in per-master-type registries and
route them the same way (see *Masters data models*).

### Masters data models

The masters data models live in `kpfpipe/data_models/masters/` — a `KPFMasterModel` base plus
the per-level `KPFMasterL1`/`KPFMasterL2` (and a `KPFMasterL4` RV/CCF stub, not yet implemented),
which subclass the science models (`KPFMasterModel` precedes them in the MRO, so its overrides win)
and so inherit the alias system and keyword machinery. Masters are *not* EPRV-governed but follow
the science keyword conventions where possible:

- **Minimal PRIMARY.** `KPFMasterL1`/`L2` stamp `DATALVL` `"ML1"`/`"ML2"` and do **not** seed the
  EPRV science skeleton (see *Header standardization*).
- **Extension schemas are CSV-driven, per master type.** `ML1-extensions.csv` builds ML1
  directly, while `KPFMasterL2(kind=…)` reads `ML2-{kind}-extensions.csv` for its required `kind`
  (`"wls"` carries `TRACE*_WAVE` + `*_WLS_COEFFS`; `"flat"` carries `TRACE*_FLUX`/`VAR`/`BLAZE`).
  Edit the CSV(s) to change a master's schema.
- **PRIMARY keywords are registered like the science models.** `MASTYPE` (every master file) and
  the WLS metadata (`ROUGHWLS`/`LINELIST`/`LINEPROF`/`POLYDEG{X,M,F}`, WLS only) are registered in
  the **per-master-type** `config/{ML1,ML2-flat,ML2-wls}-PRIMARY-keywords.csv` registries and
  unioned into the global registry table; `set_keyword` routes them as usual
  (see *Keyword registry*). `BUNIT` (on each `{chip}_IMG`) is structural, not registered.
- **QC/RECEIPT present, checks deferred.** Both levels carry `QUALITY_CONTROL` + `RECEIPT`
  extensions for later wiring, but no masters QC checks or DRP-provenance stamping exist yet.

The master filename (`{KOAID}_master_{type}_L{N}.fits`, WMKO DRP-RUN-05) is set by
`KPFMasterModel` — see *io.py path builders*.

## Processing layers

The pipeline is built in strictly one-directional layers — each layer may import *down* but
never up: `kpfpipe/` (scientist-facing building blocks) ← `recipes/` (compose modules) ←
`scripts/` (run a recipe many times) ← `tools/` (the CLI interface). So `tools/cli.py`
imports `scripts.processing.*`, but **the scripts must never import `tools`**. All four are
installed, importable packages; code shared across a layer's siblings goes **down** into
`kpfpipe/`, or — when it is layer-specific — lives beside them as a `_`-prefixed private helper
(e.g. `scripts/processing/_argparse.py`, `recipes/_logging.py`) that only its own layer imports.

### Modules

`kpfpipe/modules/` holds the scientist-facing processing primitives — each an importable
building block a recipe composes, with no orchestration or logging setup of its own. The
science modules run the *Science Pipeline* flow: `astro_query` (external-catalog
astrometry onto L0), `image_assembly` (L0 → assembled FFI),
`image_processing`, `spectral_extraction`, `wavelength_calibration`, `barycentric_correction`,
and the radial-velocity pair `radial_velocity`/`cross_correlation`; `calibration_association`
resolves which master calibrations a frame uses. The `masters/` submodule
(`bias`/`dark`/`flat`/`wls` over a shared `base.py` stacking engine) builds the calibration
products (see *Masters Pipeline*).

**Detector geometry.** Helpers like `count_amplifiers`, `orient_channels`, and `RN_KEYS` are owned by `ImageAssembly`. Other consumers (Quicklook, future Diagnostics) import them rather than duplicating the logic.

### Recipes & configs

`recipes/` compose modules into an end-to-end reduction: `kpf_drp_science.py` runs the
science flow, `kpf_drp_masters.py` builds the nightly masters. Each recipe exposes a
`main(config, args)` entry that tests call directly with no logging configured — logging
setup lives in the scripts layer, never in recipes. Default parameters live in
`configs/kpf_drp_{science,masters}.toml`.

### Scripts

`scripts/` run recipes many times, over batches of units: `processing/` holds the reduction
drivers (the CLI leaf, the orchestrators, and the timeseries wrapper — see *Command line
interface*), `plots/` the post-reduction plotter (`plot_timeseries.py`), and `quality_control/`
the reporting entry points (`qc.py`/`qlp.py`). Every driver is runnable on its own (`python -m scripts.processing.<name>`),
with no knowledge of the dispatcher above it; its flags are documented by `kpfpipe <command> --help`.

The processing drivers share a set of **`tools`-free** orchestration helpers:

- `_argparse.py` — shared argparse parent-parsers composed via `parents=[…]`, so each common flag
  (recipe/config, data dirs, logging, pool, cache) is declared once; `resolve_dir_shortcuts`
  post-parse expands the `--input_dir`/`--output_dir` convenience shortcuts into their
  per-directory slots.
- `_dispatch.py` — the process-pool engine that fans units out as subprocesses.
- `_scan.py` — the up-front, parallel-by-datecode L0 mini-db cache **pre-scan** the orchestrators
  run before fan-out (gated by `--cache`). It is deliberately the sole `kpfpipe.utils.io`
  (`FileHandler`) importer, so `_dispatch.py` stays io-free.

The default recipe/config path constants live in `kpfpipe/__init__.py`
(`DEFAULT_{MASTERS,SCIENCE}_{RECIPE,CONFIG}`) — the single source the `--masters`/`--science`
shortcuts resolve against.

**Fan-out concurrency differs by orchestrator.** Science sizes its pool from cores (`--jobs`); the
`masters` orchestrator instead caps at a **fixed** `_MASTERS_JOBS` (16, in `_dispatch.py`),
independent of `--jobs`, cores, or RAM, because masters stacking degrades with the *number* of
concurrent jobs (OS memory-mapping contention) rather than with compute or memory pressure — so do
**not** swap in a cores- or RAM-derived cap. Both orchestrators also **stagger** their subprocess
launches (`_LAUNCH_INTERVAL`): masters by 5.0 s to desync the I/O-heavy read phase each build opens,
science by 1.0 s to rate-limit the SIMBAD/Gaia catalog queries `AstroQuery` fires per frame at startup
(rationale in each module's comment). These caps and intervals were tuned empirically on
Caltech's shrek server — heuristics, not definitive values; re-confirm against a real run before
changing them.

### Command line interface (CLI)

`kpfpipe` (the `[project.scripts]` console entry, `tools/cli.py:main`) is a **thin, git-style
dispatcher**: it routes a subcommand to its implementation under `scripts/` and forwards
the remaining argv verbatim (each subcommand owns its own argparse). Full flag usage is in
`kpfpipe --help` and `kpfpipe <command> --help`; the subcommands differ by *role*:

- **`kpfpipe run`** (→ `reduce.py`) — the **leaf**: runs one recipe on one unit, in-process. Owns
  config-override assembly, `setup_logging`, the DRP-RUN-08 banner, and the recipe `exec`.
- **`kpfpipe science` / `kpfpipe masters`** (→ `science.py`/`masters.py`) — **orchestrators**: fan a
  batch of units out as one `python -m scripts.processing.reduce` subprocess each (own log, clean
  process state, independent exit) via `_dispatch.py`.
- **`kpfpipe timeseries`** (→ `timeseries.py`) — a **thin wrapper** above the orchestrators: discovers
  a target's frames from the L0 tree, then runs the masters, science, and `plot_timeseries` stages as
  subprocesses. Each stage is independently skippable and fail-soft — a frame is handed to science
  regardless of its masters result.
- **`kpfpipe plot-timeseries`** (→ `scripts/plots/plot_timeseries.py`) — the standalone **plotter**:
  renders a target's RV timeseries from its L4 products (the same stage the `timeseries` wrapper runs
  last).

## Quality control

Four layers, consolidated under `kpfpipe/quality_control/`, consume data products. None of
them mutate the scientific arrays — they only read data and write header keywords via `set_keyword`
(routed to QUALITY_CONTROL — see *Keyword registry*) (and, in Quicklook's case, to PNG files).
Per-level files follow the `levelN.py` naming used by `data_models/`.

The first three run in a strict order — **Diagnostics → QC → Checkpoints** — each consuming what the
prior wrote, driven by the recipe through a **single `CheckpointL{n}(obj).run()` call**:

- `Checkpoint.run()` folds in the paired Diagnostics and QC classes first (named on the subclass as
  the `DIAGNOSTICS`/`QC` class attributes, e.g. `CheckpointL1.DIAGNOSTICS = DiagL1`), then runs the
  checkpoint methods — so callers no longer invoke `DiagL{n}`/`QCL{n}` directly.
- The folded `QC.run()` result dict is captured on `Checkpoint.qc_results` for reporting (e.g.
  `scripts/quality_control/qc.py`). A level with no paired class skips that stage.

The recipe runs `CheckpointL0(l0).run()` **before assembly**, on purpose: QCL0 writes the L0 QC flags
onto L0's QUALITY_CONTROL, which `to_kpf1` then propagates downstream so the L1/L2/L4
products carry the full append-only QC history (e.g. `DATTIMOK`, the raw DATE-BEG/MID/END
timing-consistency flag, is an L0 check whose result rides forward this way).

This is unlike v2.12, which had one big `DiagnosticsFramework` primitive with a conditional dispatch tree over many functions and shared backend state with `AnalyzeL0/2D/L1/L2` classes. v3 uses per-level classes with method-attribute registration (`_diag_name` / `_qc_key` / `_checkpoint_name`) and no shared state.

### Diagnostics

`kpfpipe/quality_control/diagnostics/` — computes scalar/array metrics from finished data products and writes them via `set_keyword`, which routes each to its registry home — most land on QUALITY_CONTROL, but a metric registered as an EPRV PRIMARY keyword goes to PRIMARY (`DiagL2.snr` writes `SNRSC*` to QUALITY_CONTROL and mirrors the summed-SCI values to `EXSNR1-5`/`EXSNRW1-5` on PRIMARY). Per-level classes (`DiagL0`/`DiagL1`/`DiagL2`/`DiagL4`) mirror the QC structure. Examples: per-fiber NaN counts in extracted spectra, zero-flux fraction.

**Where metrics live.** Metrics that depend on intermediate processing state (e.g. read noise from raw overscan) stay in the pipeline module that produces them — they cannot be recomputed from the finished product. So does a metric a module derives from a decision it just made: `CalibrationAssociation` writes the master calibration **ages** (`BIASAGE`/`DARKAGE`/`FLATAGE`/`WLSAGE`) alongside the master paths it selects (`*FILE` on RECEIPT), so a path and its age cannot disagree. Metrics computable from the finished product alone live in Diagnostics.

### QC flags

`kpfpipe/quality_control/qc_flags/` — reads metrics (mostly from headers populated by Diagnostics or pipeline modules) and applies pass/fail thresholds. Writes **only** 0/1 keywords (via `set_keyword`, routed to QUALITY_CONTROL). No validation or raising — that is the Checkpoints layer's job.

### Checkpoints

`kpfpipe/quality_control/checkpoints/` — reads the 0/1 QC flags and the product headers and **emits warnings or raises errors** (never writes). Two inherited base checkpoints: `unregistered_keywords` (structural header validation — see *Header standardization*) and `qc_flags` (raises a failed flag named in the per-level `RAISE_FLAGS`, warns the rest) — scoped to the **current level's own** flags (`keyword_registry.qc_flag_keywords_by_level[LEVEL]`), so a propagated lower-level `0` is not re-warned. `CheckpointL0`/`L1`/`L2`/`L4` set `LEVEL` + `RAISE_FLAGS`.

### Quicklook plots

`kpfpipe/quality_control/quicklook/` — reads products and renders matplotlib plots. Pulls any annotation values from existing headers.


## Logging

Logging follows WMKO DRP-RUN-07/08/09 (issue #1408). Handler/level configuration lives in
exactly one *module*, `kpfpipe.utils.logger`, and never runs at import time or in
recipes/modules/tests. Two sibling entry points configure it:

- **`setup_logging`** — called only by the single-recipe leaf runner (`scripts/processing/reduce.py`,
  the `kpfpipe run` entry) before the recipe runs, writing that reduction's per-unit log with a
  stderr console echo.
- **`setup_batch_logging`** — a thin wrapper called once at the top of each fan-out driver's `main()`
  (the `science.py`/`masters.py` orchestrators **and** the `timeseries.py` wrapper — three callers,
  `label` ∈ `science`/`masters`/`timeseries`). It writes a `kpf_{label}_batch_{stamp}.log` of the
  *batch's own* decision points (dispatch banner, per-unit ok/FAILED, failure sentinels; for
  `timeseries`, its discovery + per-stage dispatch trail), with the console echo pinned to **stdout**
  so an operator can watch fan-out live.

The batch stdout echo is **source-filtered** (`_BatchConsoleFilter`, console handler only): below
WARNING only the driver's own narration (`scripts.*` / `__main__`) reaches the terminal, keeping
library INFO chatter out of the live view; WARNING and above always echo, and the log *file* keeps
every record. Orchestrators (and the shared `_dispatch.py` engine) narrate through named loggers,
never `print()`. Because they still fan `reduce` out as one subprocess per unit, **each reduction
also gets its own `setup_logging` per-unit log** — the batch log sits alongside, not in place of, it.

Both siblings write one UT-timestamped file per invocation under the `[LOGGER] log_dir` config key
(`log_level`/`console` also honored; CLI `--log_dir`/`--log_level` override); a missing `log_dir` is
fatal (DRP-RUN-07). Library code only declares `logger = logging.getLogger(__name__)` and must work
with no handlers installed — tests call `recipe.main(config, args)` directly with none configured, so
setup must never move into recipes. Recoverable/degraded conditions use `logger.warning` (not
`warnings.warn`); `logging.captureWarnings` still funnels any third-party/stdlib `warnings.warn` into
the log at `WARNING`. Tests that configure logging must tear down with `teardown_logging`
(see the autouse fixture in `tests/regression/test_logger.py`).


## File handling

### Data directory structure

The pipeline reads and writes a small number of on-disk trees, rooted at the `[DATA_DIRS]`
config keys — `KPF_DATA_INPUT` (the raw L0 input tree), `KPF_MASTERS_OUTPUT` (the masters
output tree), and the science/QLP output roots. `kpf_directory` (see *io.py path builders*) is
the single authority for the output layout:

- **L0 input**: `{KPF_DATA_INPUT}/L0/{datecode}` — the raw `KP.*` frames, alongside the
  data-tree artifacts `{KPF_DATA_INPUT}/vNext/reference/junk_obs.csv` (the junk list) and
  `{KPF_DATA_INPUT}/vNext/mini_db/{datecode}_L0.csv` (the cached per-night mini database).
- **Science products**: `{data_root}/{level}/{datecode}` for `L0`/`L1`/`L2`/`L4`.
- **Masters**: `{KPF_MASTERS_OUTPUT}/masters/{datecode}`.
- **Quicklook**: `{data_root}/QLP/{datecode}/{obs_id}/{level}`.

### FileHandler class

`kpfpipe/utils/io.py::FileHandler` discovers KPF files across the L0-input and masters-output
trees, keyed by `datecode`/`cal_type`, so recipes and scripts never assemble data paths by
hand. It is constructed from the already-extracted `[DATA_DIRS]` mapping (it does not import
`ConfigHandler`, keeping construction light). Its key methods:

- `build_mini_database(datecode)` — scans a night into a per-frame table, tagging each frame
  with the derived **`ISJUNK`** column (from `junk_obs.csv`; used by masters stacking to exclude
  junk) and caching the result to disk.
- `build_calibration_stacks(...)` — groups frames into per-`cal_type` stacks for master
  building; `exclude_junk=True` filters junk out first.
- `find_masters(cal_type, level, datecode)` — locates already-built master calibrations.

It is **not thread-safe** (`build_mini_database` stores the scanned night on the instance), so
use one instance per thread; the on-disk mini-db cache is shared safely, keyed by datecode.

### io.py path builders

Science filenames: `L0 = {obs_id}.fits` (KPF-native `KP.*`); `L1 = kpf_L1_{YYYYMMDD}T{HHmmss}.fits` — note **no EPRV "S"**, because the EPRV standard defines no L1 (its filename regex only accepts `SL2`/`SL3`/`SL4`); `L2/L4 = kpf_SL{N}_…` (EPRV-standard). Masters (WMKO DRP-RUN-05): `{KOAID-of-first-input}_master_{type}_L{N}.fits`.

Two authorities encode these names and **must agree per level**: `kpf_filepath()` (`utils/io.py`) —
the pipeline's path builder (output directory + basename, from an obs_id string), what recipes use to
write — and `<model>.generate_standard_filename()` — basename only, the `to_fits(fn=None)` fallback.
`kpf_filepath` composes two single-authority helpers: `kpf_directory` (the sole authority for an
**output** directory — the `science`/`masters`/`QLP` layouts above) and `kpf_filename` (**the sole
authority for the naming rule**). All four science models' `generate_standard_filename` delegate to
`kpf_filename(self.obs_id, level)`, so the object- and string-keyed builders cannot drift — which is
why every model must carry `obs_id` on every construction path (see the `ORIGID` note under *Header
standardization*). As with `check_filename_convention`, **every concrete model declares
`generate_standard_filename` explicitly** (masters override it with the KOAID/MASTYPE name, carrying
no `obs_id`) while `KPFDataModel`'s base version **raises `NotImplementedError`**;
`TestFilenameConsistency` (`tests/regression/test_io.py`) guards the two builders against drift.
Filename *validation* (`check_filename_convention`) is separate, delegating to rvdata's EPRV check
for L2/L4.

## Tests

`tests/` splits into `regression/` (the pytest suite) and `profiling/` (performance harnesses),
with `tests/conftest.py` at the root serving both (its fixtures and the `requires_testdata`
marker) and the real frames under the gitignored `tests/testdata/`. This section covers how the
suites are **laid out**.

### Regression

`tests/regression/` holds `test_<module>.py` files mirroring the source (per-level names follow
`data_models/`, e.g. `test_quicklook_l0.py`), plus the non-collected helpers `_masters.py`
(synthetic fixtures) and `_dtype_policy.py` (the dtype rubric). The `slow` marker carves the
real-`testdata` integration and heavy-compute tests (full L0→L2, real-frame assembly/overscan,
master stacking, WLS orientation) off from the fast `-m "not slow"` subset.

The masters tests mirror the masters subpackage by *responsibility*: `test_master_base.py` covers
the shared stacking engine (`BaseMasterModule`), `test_master_bias.py`/`test_master_dark.py` the
concrete bias/dark modules, `test_master_wls.py` the WLS path (a separate ML2),
`test_masters_recipe.py` the `kpf_drp_masters` recipe, and `test_masters_script.py` the masters
CLI script; `flat` has no test file while stubbed.

### Profiling

`tests/profiling/` harnesses mirror the test files 1-to-1 (`profile_<x>.py` ↔ `test_<x>.py`),
share logic in `_profiling.py`, and write reports to the gitignored `tests/profiling/reports/`.
Two recipe harnesses (`profile_science_recipe.py`, `profile_masters_recipe.py`) rank functions
*across* modules to show which stage dominates; per-module `profile_<module>.py` files drill into
one module. `flat` is skipped while stubbed, and the shared stacking engine (`masters/base.py`)
has no dedicated harness — its work is attributed to `base.py` methods inside the bias/dark
harnesses (whereas the *test* suite isolates the engine in `test_master_base.py`, because
profiling partitions by wall-clock and tests by responsibility).
