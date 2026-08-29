# KPF-DRP vNext: Code Style Guide

This guide captures the coding conventions actually in use across the KPF-DRP vNext
codebase, so that new code (whether written by a developer or by Claude) reads like
the code already here. It was derived by surveying every subpackage — standard
modules, masters, quality control, utils, recipes/configs, tests, and documentation.

**Authority precedence.**
When requirements or design principles conflict, the order of governing document precedence is:

1. WMKO technical requirements ([`WMKO_REQUIREMENTS.md`](WMKO_REQUIREMENTS.md))
2. EPRV data standard ([`EPRV_DATA_STANDARD.md`](EPRV_DATA_STANDARD.md))
3. KPF vNext project charter ([`KPF_VNEXT_CHARTER.md`](KPF_VNEXT_CHARTER.md))
4. KPF vNext architecture reference ([`KPF_VNEXT_ARCHITECTURE.md`](KPF_VNEXT_ARCHITECTURE.md))
5. KPF vNext style guide ([`KPF_VNEXT_STYLE_GUIDE.md`](KPF_VNEXT_STYLE_GUIDE.md))

When any two conflict, the higher one wins.


**Status of these rules.** These are *soft* requirements — this guide sits at the **bottom** of
the authority hierarchy above, so where a rule here conflicts with any of those four documents,
**the higher one wins** (style yields to science). It describes the dominant, prevailing pattern;
where the codebase contradicts itself, the recommended variant is called out explicitly.

This file covers *how code should look and be organized* — not operational/technical guidance
(environment, commands) or the pipeline's structure (the architecture reference).

Contents:

* **A. Core Design Principles**
* **B. Pipeline design**
  * B.1 KPF data models — Science, Masters
  * B.2 KPF modules — Science, Masters
  * B.3 Recipes & configuration — Science, Masters
  * B.4 FITS structures — Extensions, Headers, Keywords
  * B.5 Quality control — Diagnostics, QC, Checkpoints, Quicklook
  * B.6 Error handling, validation & logging
* **C. Coding conventions**
  * C.1 Naming
  * C.2 Class design
  * C.3 Function & method design — Type hints
  * C.4 NumPy & numerical idioms
  * C.5 Shared utilities & helpers
  * C.6 Imports
  * C.7 Formatting
  * C.8 Tests — Regression, Profiling
* **D. Documentation**
  * D.1 Docstrings
  * D.2 Comments
* **E. Open Inconsistencies**

---

## A. Core Design Principles

These are the values the rest of this guide operationalizes. Each principle names where it is
enforced concretely.

- **Code is self-documenting.** Names carry intent, so a reader rarely needs a comment to learn
  *what* a line does; comments are reserved for *why*. This is the lever behind the naming rules
  (§C.1) and the comment discipline (§D.2).
- **Prefer clarity over cleverness.** Write the obvious version. A longer, plainly-readable
  implementation beats a terse or clever one that a future reader — human or AI assistant — has
  to decode.
- **Do not over-engineer.** Implement the simplest thing that meets the requirement; don't build
  for hypothetical futures (YAGNI). A new utility earns its place with a real caller (§C.5).
- **Avoid excessive abstraction.** Keep layers minimal. Transform modules are plain standalone
  classes — no base classes, mixins, ABCs, or `dataclasses` unless a concrete, shared need
  already exists (§C.2).
- **No hidden state.** No implicit global state, no database coupling in science code, no
  calibration hierarchy resolved behind the caller's back. Every lazily-filled attribute is
  declared up front in `__init__`, and configuration flows through one explicit three-tier
  override chain (§C.2).
- **Fail loudly.** Validate early and raise a specific, typed exception that shows the offending
  value; never swallow an error or press on with degraded data. Raise rather than catch-and-log
  (§B.6).
- **Do not introduce quiet fallbacks.** No silent retries, no "default on missing key" that masks
  a real problem — a missing keyword or master should raise, not be papered over (§B.6).
- **Be explicit.** Explicit calibration paths, explicit units and reference frames, explicit
  arguments. Every step should be readable and debuggable in isolation.

---

## B. Pipeline design

Conventions organized by pipeline subsystem. Where a subsystem has distinct **science** and
**masters** flavors, they are called out under their own subheadings.

### B.1 KPF data models

The models (`data_models/`) are what modules operate *on*. Conventions specific to writing model
code:

#### Science

- **Instance vs. level/manifest naming** (`kpfN` vs `L<N>`). A variable *holding a model
  instance* uses `kpfN` in `data_models/` (`kpf2 = self.to_kpf2()`) and `lN_obj`/`mlN_obj` in
  `modules/` (`l1_obj`, `ml2_obj`). A constant/identifier naming a data **level**, extension
  **manifest**, or **filename pattern** uses the `L<N>`/`ML<N>` token everywhere
  (`_L0_EXTENSIONS`, `_L1_TO_L2_PASSTHROUGH`, `_ML1_EXTENSIONS`), mirroring the EPRV `DATALVL`
  terminology and the CSV names. A receipt **step-name string** matches its *method*
  (`to_kpf1`/`to_kpf2`/`to_kpf4`), not the level.
- **I/O overrides forward KPF's conventions**: `to_fits` keeps a single positional filepath
  (forwarding it as rvdata's renamed `out_filename`); materialize memmapped HDU arrays with
  `np.array(hdu.data)` (not `np.asarray`) before the source file closes, so nothing aliases a
  freed memmap.

#### Masters

- Master models (`data_models/masters/`) subclass `KPFMasterModel` and use the `ML<N>` token in
  their manifests (`ML1-extensions.csv`, `_ML1_EXTENSIONS`).

### B.2 KPF modules

#### Science

Every standard pipeline module follows the same top-to-bottom template. New modules should
match it:

```python
"""KPF <Stage> module.

<Longer modules add a multi-paragraph description, dispatch tables, and a
Notes section citing papers — see radial_velocity.py / barycentric_correction.py.>
"""

import os                                   # 1. stdlib
import numpy as np                          # 2. third-party
from kpfpipe import DEFAULTS                 # 3. first-party (absolute only)
from kpfpipe.utils.config import ConfigHandler

_DEFAULTS = {**DEFAULTS, "module_param": ...}   # module constants are private (leading _)


class StageName:
    """One-line summary.

    Parameters
    ----------
    l1_obj : KPF1
        <constructor args documented here, at the CLASS level>
    config : None | dict | ConfigHandler
        ...
    """

    def __init__(self, l1_obj, config=None):
        self.l1_obj = l1_obj
        # ... canonical config block (see §C.2) ...
        self._info = None  # cached info() summary text (str)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _helper(self, ...): ...

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------
    def do_step(self, ...): ...

    # ------------------------------------------------------------------
    # Private helpers - module execution
    # ------------------------------------------------------------------
    def _track_info(self, chips=None, fibers=None):
        """Build & cache the info() summary text (takes only chips/fibers)."""
        self._info = "\n\n" + "\n".join(lines) + "\n\n"   # blank-line padding lives here

    def _set_headers(self, l2_obj):
        """Sole place this module writes headers; reads instance attributes."""
        l2_obj.set_keyword(KEY, self._attr)    # routed to its registry extension

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def perform(self, chips=None):
        ...                                    # populate header-source attributes
        self._set_headers(self.l2_obj)         # consolidates ALL header writes
        self._track_info(chips)                # caches _info text, before the receipt
        self.l2_obj.receipt_add_entry("stage_name", "", "PASS")
        logger.info("%s", self._info)         # padding is baked into _info; keep this clean
        return self.l2_obj

    def info(self):       # prints self._info (the cached text), always last
        print(self._info)
```

- **Method order is fixed, marked with 66-dash banner comments** (`# ----`, never `====`/`####`;
  sub-labels allowed): `__init__` → *Private helpers* → *Algorithm steps* → *Private helpers -
  module execution* (`_track_info`, then `_set_headers`) → *Public entry point* (`perform`) →
  `info()`.
- **Header consolidation & `_info`.** A module writes headers in exactly one place — a private
  `_set_headers(obj)` sourcing values from instance attributes (never recomputing, never reading
  another product), each via `obj.set_keyword`, called just before `receipt_add_entry`; a module
  that writes none keeps an empty helper. `_info` is the cached human-readable summary **text** (a
  `str`) for reporting only — never the science/header chain, never asserted by tests.
  `_track_info(self[, chips, fibers])` builds and caches it on `self._info`; `perform()` logs it
  and `info()` prints it (one shared rendering), and `info()` prints a "not been called" notice
  while `_info` is `None`.
- **Porting from the legacy DRP** (`legacy/KPF-Pipeline/`, an algorithm reference): read the
  original `modules/<stage>/src/alg.py` for the core logic, then rewrite it to the skeleton above
  — do **not** carry over hidden state, DB dependencies, or implicit calibration hierarchy.

#### Masters

The masters layer (`modules/masters/`) is a batch/stack builder that diverges from transform
modules — follow *its* conventions:

- **Constructed from a file list**, not a data object: `Bias(l0_file_list, config)`.
- **Entry points are `make_master_l1/l2(...)`**, not `perform()`, with the same positional-then-
  keyword-only shape as `perform()` (§C.3) and `l0_file_list` as the sole positional.
- **Thin subclasses over `BaseMasterModule`.** `Bias`/`Dark`/`Flat`/`WLS` add only an `__init__`
  (selecting the config section), the `make_master_*` entry point, and an `info()` ASCII-table
  summary; `_DEFAULTS` extends the base (`{**BaseMasterModule._DEFAULTS, ...}`).
- **Public API is `make_master*`, `save*`, `stack_frames`, `info`**; every other method is
  underscored (tests patch the underscored names directly).

### B.3 Recipes & configuration

#### Science

- **Recipes are plain Python** with one `main(config, args)` entry (tests call it directly); no
  top-level execution. Their composition and the scripts/CLI layering are in the architecture
  guide.
- **Invoke a module by instantiate-then-call**, the variable named `snake_case` after the class:
  `image_assembly = ImageAssembly(l0, config); l1 = image_assembly.perform()`. Transforms →
  `.perform()`, QC/Diag/Quicklook → `.run()`, masters → `.make_master_l1/l2()`. The constructor
  takes the whole `ConfigHandler`; each module pulls its own section.
- **Build output paths only through the `utils/io.py` builders** (`kpf_directory`/`kpf_filename`/
  `kpf_filepath`) — never string-concatenate paths in modules/recipes; `os.makedirs(
  os.path.dirname(path), exist_ok=True)` before every `.to_fits()`. (Plain `os.path.join` is fine
  for night-keyed *input* directories that have no obs_id.)
- **Validate args at the top**: `raise SystemExit("Error: --obs_id is required …")` with an example.
- **Recipe comments** are terse, lowercase, imperative, and explain the *why*, above the stage
  (transforms annotated `-->`); disabled stages stay as commented-out blocks labelled
  `(not yet implemented)`.

Config files:

- **TOML via stdlib `tomllib`**, one per recipe sharing the recipe basename. Sections are
  `UPPER_SNAKE`: the fixed `[DATA_DIRS]`/`[TRACES]` globals plus per-module `[MODULE_<NAME>]`
  matching the class, pulled via `config.get_params([...])`. Keys are `lower_snake_case` (except
  `[DATA_DIRS]`, which is `UPPER_SNAKE`); floats carry an explicit `.0` in TOML arrays.

#### Masters

- **Masters config sections use the bare product name** — `[BIAS]`/`[DARK]`/`[FLAT]`/`[WLS]` (no
  `MODULE_` prefix). A deliberate, accepted exception — don't "align" it to the science pattern.
- **Frame-calibrating masters merge the shared transform sections** they reuse — Dark/Flat/WLS
  pass `[..., "<PRODUCT>", "MODULE_CALIBRATION_ASSOCIATION", "MODULE_IMAGE_PROCESSING"]` with
  `MODULE_IMAGE_PROCESSING` **last** so its flags win on any collision; Bias omits both.

### B.4 FITS structures

*(Terminology: **`rvdata`/`RVData`** = the Python package we import and inherit from; **"EPRV"** =
the data format/spec it implements. Keep them distinct — never write "rvdata standard" for the
format; say "EPRV standard (which rvdata implements)". Identifiers holding format artifacts take an
`eprv_` prefix. One quirk: rvdata's vendored `header_map.csv` names its target column `STANDARD`,
but its values are EPRV targets.)*

#### Extensions

- **CSV config tables** (`data_models/config/`) drive HDU layout and alias registration: comma-
  separated, single header row, read with `pandas.read_csv`, always ending in a `Description`
  column. Extension manifests (`L0-extensions.csv`, …) are `HDU,Name,DataType,Required,Description`;
  mapping tables are `aliases.csv` (`KPF,EPRV,…`) and `trace-map.csv` (`Trace,Fiber,…`).
- **Keep fiber names in sync** across `trace-map.csv`, `[TRACES].fibers`, and `detector.toml`.

#### Headers

Every extension header is an `astropy.io.fits.Header`. When writing code:

- **Read a value** with `header.get(key)`/`header[key]` on the keyword's home extension (per the
  registry `Extension` column); the comment is in `header.comments[key]`. Never hand-roll
  `value[0] if isinstance(value, tuple)` — headers are never tuple-valued.
- **Write a registered keyword** with `obj.set_keyword(key, value)` — it routes to the registry
  home with the registry `Description` as comment. Never hardcode an extension/comment or write
  `headers["PRIMARY"][key] = …` for a registered keyword; the keyword must be in
  `config/{prefix}-{EXTENSION}-keywords.csv` first, or `set_keyword` raises. Never write to
  `INSTRUMENT_HEADER` outside `StandardizeDataFormat`.
- **The native → EPRV conversion has one home**, `StandardizeDataFormat`; never re-implement it,
  and never read a native card off PRIMARY once it has run.
- **Prefer PRIMARY, fall back to `INSTRUMENT_HEADER`** for reads: a native that survives on PRIMARY
  under its own name is read there; one that never reaches PRIMARY, or that reads more clearly as a
  coherent native block, is read from `INSTRUMENT_HEADER`. No silent fallback — let a missing key
  raise.
- **Use EPRV keyword names on PRIMARY** (`EXPTIME`, not `ELAPSED`); the L4 `RV{GREEN,RED}`/`ERV{GREEN,RED}` are
  the one KPF-registered exception deliberately homed on PRIMARY.

#### Keywords

- **Keyword names**: ≤8 chars, uppercase, no underscores (`NANSCI1`, `ZEROSCI1`); encode the level
  where needed for uniqueness (`DATAPRL0`). **Before coining a new keyword, reuse the legacy
  spelling where the science meaning matches** (`WLSFILE`, `BIASFILE`), so downstream/archival
  tools keep reading v3 products; `reference/legacy_data_format.rst` is no longer vendored, so
  read it out of git history.
- **Register every KPF keyword in `config/{prefix}-{EXTENSION}-keywords.csv`**
  (`Keyword,Description,Units,DataType,ExampleValue,PopulatedBy`) — the *filename* names the home
  extension, and `Description [Units]` becomes the FITS comment (both defined once, here).
  `ExampleValue` is documentation only and is never read by the pipeline; a blank `PopulatedBy`
  means nothing writes an informative value to the keyword yet. Flags are stored as `int` 0/1
  (never Python bool): QC keys get a `QC: …` description, other flags append `(T/F)`. `#` is the
  one template marker and expands to `1..DETECTOR["numtrace"]`, in a keyword (`CRA#`) or in a
  filename's family stem (`L2-TRACE_WAVE`); it is reserved, so the CSVs carry no comment rows.
  Keep `Description [Units]` inside 47 characters or astropy truncates the card.

### B.5 Quality control (Diagnostics / QC / Checkpoints / Quicklook)

The four layers live in `kpfpipe/quality_control/`. Conventions for writing QC code:

- **Never mutate `data`.** Diagnostics/QC write header keywords **only via `set_keyword`** (→
  QUALITY_CONTROL), never `data`; Quicklook writes only PNGs. To mutate `l0.data` in a helper, work
  on a `deepcopy`.
- **Register methods by attribute, no decorators** — assign the tag right after the `def`; the base
  `_iter_*` generator walks the MRO (subclass beats base):
  ```python
  def nan_counts(self): ...
  nan_counts._diag_name = "nan_counts"   # Diagnostics (_qc_key / _checkpoint_name for QC / Checkpoints)
  ```
- **Runners reset `self.results = {}` at entry** and wrap each method in `try/except`, re-raising as
  `RuntimeError` (loud failure, no silent suppression).
- **QC writes `int` 0/1 and does no validation**; round floats
  (`round(float(x), 6)`) and cast numpy scalars to Python types. The per-check comment lives once in
  the registry `Description` (QC methods carry only `_qc_key`). QC comments are namespaced `"QC: …"`;
  Diagnostics comments are bare phrases.
- **Quicklook** follows the existing plot conventions (pyplot state-machine, `viridis`, percentile
  vmin/vmax, templated titles + `_zoomable.png` filenames), stubbing unimplemented plots with
  `NotImplementedError` citing the v2.12 source. Its `run()` contract — save-and-close when
  `output_dir` is set, else return the figures **open** — is stated once here; per-level `run()`
  docstrings reference it rather than restating it.

### B.6 Error handling, validation & logging

Logging follows WMKO DRP-RUN-07/08/09. The coding rules:

- **Named loggers only**: `logger = logging.getLogger(__name__)` at module top; recipes (exec'd, so
  `__name__ == "recipe"`) name theirs explicitly (`"kpfpipe.recipe.science"`). Never the root-logger
  conveniences (`logging.info(...)` — Ruff `LOG015`).
- **Handler/level configuration lives only in `kpfpipe.utils.logger`**, never at import time or in
  recipes/modules/tests; library code must work with no handlers installed.
- **Level policy**: `INFO` = production (steps, decisions, I/O, end-of-`perform()` summaries),
  `DEBUG` = inner-loop (per-order/-chip/-frame) detail, `WARNING` = recoverable/degraded runtime
  conditions. Per-item logging inside a loop stays `DEBUG` (or aggregate to one line) so it never
  floods `INFO`/`WARNING`. Emit QC/diagnostics keyword logs one keyword per line as
  `keyword = value — comment` (the FITS comment, read off the header) — verbosely at `DEBUG`, but
  only for failing/fatal flags at the rare `WARNING`/`ERROR` levels. Never gate log calls behind
  `verbose` — the level is the gate. Use lazy
  `%`-formatting (`logger.info("wrote %s", fn)`, Ruff `G`), not f-strings.
- **No `print()` in pipeline code.** Anything that runs as part of a reduction — `perform()` and its
  helpers, recipes, the CLI, utils, data-model read/write paths — logs, never prints. The **only**
  sanctioned `print()` is the interactive `info()` reporter (data models and modules), which exists
  for notebook/REPL use and is never called from pipeline paths (the `_info` rendering is in §B.2).
- **Raise; don't catch-and-log.** The sole sanctioned catch-log-reraise point is the leaf runner
  (`reduce.py`: `logger.critical(..., exc_info=True); raise`). Elsewhere pick the semantic type:
  `TypeError` (wrong `config` type), `ValueError` (bad domain value — the workhorse),
  `LookupError`/`KeyError` (missing trace/header), `FileNotFoundError` (missing master/input),
  `NotImplementedError` (stubs), `RuntimeError` (wrapping a sub-step failure).
- **No `assert` for validation** — explicit `if …: raise`. Validate early, at the top of the
  function. Error messages are f-strings that state the expectation and show the value with `!r`.
- **Narrow the `except`**, never bare. Parenthesize multi-type clauses (`except (ValueError,
  TypeError):`) — required because `target-version` is pinned `py313` (§C.7). Always chain re-raises
  (`raise … from e`, or `from None` when translating a low-level error into a clearer domain one).
- **Predicate/extractor split**: `is_*` predicates validate inline and return `bool` (never raise);
  the matching raising extractor/converter (`get_*`, `utc_to_hst`, …) validates through the predicate
  and raises `ValueError` at its own boundary.
- **Recoverable/degraded conditions** → `logger.warning(...)`, not `warnings.warn`. A degraded
  runtime condition (missing header, dropped frame, failed lookup) is an operational event the log
  must record (DRP-RUN-08), and `logger.warning` — unlike `warnings.warn` — does not dedup per
  call-site, so a per-frame condition is logged every time. Reserve `warnings.warn` for
  developer-facing deprecations / API misuse. `setup_logging` still calls
  `logging.captureWarnings(True)`, so any third-party/stdlib `warnings.warn` is funneled into the log
  at `WARNING`.

---

## C. Coding conventions

Cross-cutting conventions that apply regardless of subsystem.

### C.1 Naming

- **Names carry intent**, so a reader rarely needs a comment for *what* a line does (the lever
  behind the §D.2 why-not-what rule). Keep sanctioned terse notation — documented algorithm symbols,
  domain short names (`chip`, `order`, `oscan`) — where it aids readability. Below are the
  KPF-specific conventions (beyond the PEP 8 defaults ruff already enforces):

| Thing | Convention | Examples |
|---|---|---|
| Masters modules | short, single-word = the product | `bias.py`, `dark.py`, `wls.py` |
| Acronym classes | keep the acronym capitalized | `WLS` (not `Wls`), `QC` |
| Per-level classes | compact level suffix `L0/L1/L2`, **not** `Level0` | `DiagL0`, `QCL1`, `KPF2` |
| Predicates | `is_*`, return `bool` | `is_obs_id`, `is_timestamp` |
| Converters | `<x>_to_<y>` | `air_to_vac`, `utc_to_hst` |
| Public constants | `UPPER_SNAKE`; allowed in `data_models/` and the package root, **never in `modules/`** | `DEFAULTS`, `DETECTOR`, `NORDER_GREEN` |
| Module constants | `_UPPER_SNAKE`; modules export **no** importable constants (lone exception: `ImageAssembly.RN_KEYS`) | `_DEFAULTS`, `_OBS_ID_PATTERN` |

- **One public class per module**, its name = CamelCase of the filename.
- **Math-heavy locals may be terse single capitals** mirroring a published algorithm (Horne 1986
  `D, V, S, F, P, M, W`) — only in numerical code, only when documented in the surrounding
  docstring. Use descriptive names everywhere else.
- **Modules define no public constants.** Every `kpfpipe/modules/` module-level constant is
  `_`-private; pull detector geometry from `DETECTOR` (exposed on every instance as `self.norder`
  etc.) and physical constants from `astropy` rather than re-declaring them.

### C.2 Class design

- **Transform modules are plain standalone classes** — no base class, mixins, ABCs, or
  `dataclasses`; they operate *on* data-model objects, not subclass them. (The masters and QC/Diag
  layers *do* share a base class — §B.2/§B.5.)
- **Canonical constructor**: `__init__(self, l<N>_obj, config=None)` — the data object first, named
  for its level.
- **Copy the config-resolution block verbatim**, then let `_DEFAULTS` become instance attributes:
  ```python
  if config is None:
      params = {}
  elif isinstance(config, dict):
      params = config
  elif isinstance(config, ConfigHandler):
      params = config.get_params(["DATA_DIRS", "TRACES", "MODULE_<NAME>"])
  else:
      raise TypeError("config must be None, dict, or ConfigHandler")
  for k, v in _DEFAULTS.items():
      setattr(self, k, params.get(k, v))
  ```
  `_DEFAULTS` merges the globals (`{**DEFAULTS, ...}`). Resolution is a three-tier chain, lowest
  first: `_DEFAULTS` (in-module default) → config (TOML) → a direct method kwarg (the developer/
  interactive override, not used in production).
- **Declare every lazily-populated attribute in `__init__`** (to `None`/`{}`), with a trailing
  `# populated by …` comment only where the filler isn't obvious (no hidden state):
  ```python
  self._ccd_bjd = None   # per-CCD [GREEN, RED] arrays for _set_headers
  self._line_mask = {}   # set by _build_line_mask()
  ```
- **Dispatch-by-name** for pluggable methods: `getattr(self, f"_{method}_extraction")` wrapped in
  `try/except AttributeError → raise AttributeError("Unsupported …")`.

### C.3 Function & method design

- **"`None` means use config"** — the most universal convention in the codebase. Domain identifiers
  (`chip`, `fiber`, `order`) are positional; every tunable is a keyword defaulting to `None` and
  resolved at the top of the body (`if chips is None: chips = self.chips`).
- **Canonical `perform()` signature**: `perform(self, chips=None, fibers=None, *, <kwargs>)`.
  `chips`, `fibers` are the only positionals (omit either if unused; a module whose primary selector
  is something else keeps it in the same slot). Everything else is keyword-only after a bare `*`, in
  two groups: **configurable** params (`=None`, resolving to `self.<attr>`) first, then **semi-hidden**
  knobs with a real literal default (e.g. `min_npts=9`) that are absent from `_DEFAULTS`/config. The
  tier must be legible from the signature — `=None` ⇒ configurable, literal ⇒ semi-hidden — so a
  semi-hidden sequence default is an immutable tuple (`clip_edge_pixels=(500, 500)`), never a
  `None`-sentinel + in-body list. `make_master_*` follows the same shape with `l0_file_list` as the
  positional (§B.2).
- **This ordering applies to every method**, not just entry points; the `*` itself is reserved for
  the public entry points (`perform`/`make_master_*`) — private methods keep the ordering without a `*`.
- **String-enum mode params** (`method`, `cal_type`, …) validate against an explicit allowed set,
  raising `ValueError` that names the options.
- **Return patterns**: a transform's `perform()` returns the next-level object after
  `receipt_add_entry("<module>", "", "PASS")` (the middle arg is the rvdata `ARGS` string, `""` when
  N/A); in-place step methods return `None` and document *"Modifies … in-place"*; helpers return a
  single value or a fixed tuple.
- **Never mutate inputs** in utils/numerical code — `.copy()` (or `np.asarray(x).copy()`) first.

#### Type hints

- **Docstring types only — no PEP 484 annotations**; the codebase carries none, so don't start.
  `mypy` is a vestigial dev dependency, not run or enforced.

### C.4 NumPy & numerical idioms

- **Vectorize**; use explicit broadcasting (`[:, None]`, `[None, :]`) with named temporaries. Loops
  only for inherently sequential work.
- **Be NaN-aware by default**: `np.nanmedian`, `np.nanstd`, `np.nanmean`; fill missing data with
  `np.full(..., np.nan, dtype=np.float32)`.
- **Dtype precision is a contract — guard both directions.** Never upscale `float32→float64`
  (memory/throughput regression) nor downscale `float64→float32` (precision loss → wrong RVs). The
  policy — single source of truth, also encoded for tests in `tests/regression/_dtype_policy.py`:
  - **float32** — L1 `*_CCD`/`*_VAR`, master `*_IMG`/`*_SNR`, L2 `*_FLUX`/`*_VAR`/`*_BLAZE`.
  - **float64** — every `*_WAVE`, `BJD_TDB`, `BARYCORR_KMS`/`_Z`, CCF cubes, and the L4 RV-table
    floats. `*_WAVE`, `BJD_TDB`, `WAVE_START`/`WAVE_END` are **EPRV-mandated 64-bit** (born-64 at
    every state — never rely on RVData's upcast); the rest is KPF precision policy.
  - **bool** in memory / **uint8** on disk — quality masks (`*_MASK`).
  - L0 amps stay native-int or float32 — **never float64**.
  Be explicit at allocation (`np.zeros(..., dtype=...)`) and cast kernels/weights to the input dtype
  so scipy doesn't promote. A deliberate change that yields a higher-precision *result* (a float64
  CCF from float32 flux) is fine — the result's dtype governs.
- **Prefer robust statistics**: median + MAD (`astropy.stats.mad_std(..., ignore_nan=True)`) over
  mean/std for outlier work; guard divisions with a small `eps` (`1e-12`) or `np.maximum(N, 1)`.
- **Pre-zero then fill valid pixels** rather than divide-then-clean; use the `where=` kwarg of
  `np.sum` et al. for masked reductions.
- **Views vs copies are deliberate**: slicing yields views on purpose; `.copy()` when you must mutate.
- **Row/col nomenclature is numpy, not KPF.** All image/spectrum arrays use **axis 0** (`row`/`nrow`)
  = **cross-dispersion** and **axis 1** (`col`/`ncol`) = **dispersion** — the *transpose* of the
  KPF/observatory physical convention, so a reader expecting KPF physical directions will misread
  `row`/`col`. The code is uniform and self-consistent; annotate the axis convention where it matters.

### C.5 Shared utilities & helpers

- **Utils-first: import shared helpers, don't duplicate.** Reusable stats/validation/geometry live in
  `kpfpipe/utils/` (detector geometry — `count_amplifiers`, `orient_channels`, `RN_KEYS` — on
  `ImageAssembly`) and are imported, never re-implemented. `scipy` is the numerical backend; shared
  numerics (`flag_outliers`, `optimize_lsq`, `interpolate_bad_pixels`, `compute_redshift`) live in
  utils.
- **A util earns its place with a caller (YAGNI).** Two kinds may be kept without a call site, **each
  saying so in its docstring**: the symmetric-completeness half of an inverse/validator pair
  (`hst_to_utc`), and a staged-ahead helper already covered by tests (`air_to_vac`). Anything else
  unused is dead code.
- **A `utils/` IO/discovery handler is a lighter class variant** (e.g. `io.FileHandler`): it takes the
  already-extracted `[DATA_DIRS]` dict (not a `ConfigHandler`) and omits the `_DEFAULTS` loop when
  nothing is tunable, keeping per-call knobs as method arguments.

### C.6 Imports

- **Absolute imports only** — never relative (`from kpfpipe.modules.masters.base import …`, not
  `from .base import …`). Grouping and sorting are ruff-enforced (`I`); standard aliases are `np`,
  `pd`, `u` (astropy.units), `plt`.
- **Deferred (in-function) imports** are acceptable to break import cost/cycles — add a one-line
  comment saying why.

### C.7 Formatting

- **`ruff format` and `ruff check` own formatting and linting** (black-compatible: 88 cols, double
  quotes), configured in `pyproject.toml` and enforced by the pre-commit hook. Follow the formatter
  rather than hand-styling against it.
- **`target-version` is pinned `py313`** (one below the 3.14 runtime) on purpose: at `py314`,
  `ruff format` strips the parens off a multi-type `except`, emitting PEP 758's bare form, which
  Pylance/Pyright can't parse. The pin keeps `except (A, B):` (§B.6).
- **f-strings are the only interpolation style** — no `%` or `.format()`, except `strftime` codes,
  deliberate numeric formatting (`format(x, "g")`), and the lazy `%` in log calls (§B.6).

### C.8 Tests

#### Regression

- **pytest, class-based** — `Test<Subject>` classes, no bare module-level `test_` functions; methods
  `test_<behavior>`, error paths suffixed `_raises`. Files `test_<module>.py` mirror the source,
  sectioned with the same 66-dash banners and opened with a scope-stating module docstring.
- **Masters test placement**: a test belongs in `test_master_base.py` iff it exercises a `base.py`
  method vehicle-incidentally; module-specific behavior stays in `test_master_<type>.py`.
- **Fixtures** are named for what they produce; multi-file fixtures live in `tests/conftest.py` and
  multi-file plain helpers in a non-collected `_underscore.py` module — don't duplicate builders. Use
  `scope="class"` + `tmp_path_factory` for expensive real-data pipelines.
- **Test data**: real FITS under the **gitignored** `tests/testdata/<LEVEL>/<date>/`, via
  `Path(__file__).parent / "testdata" / …`. Two tiers (synthetic in-memory vs. real `testdata/`)
  documented in the module docstring. Never commit testdata and don't build a fixture-generation
  script (there is none, by design); regenerate a missing frame locally and flag it in the response.
- **Markers**: `@pytest.mark.slow` (integration/heavy) and `@pytest.mark.requires_testdata`
  (auto-skipped when absent).
- **Tolerances**: analytic recovery `assert_allclose(rtol=1e-5, atol=1e-5)`; FITS round-trips
  `assert_array_almost_equal(decimal=4)`; scalars `pytest.approx`; real-data sanity bare ranges.
- **Assertions**: bare `assert` for scalars/shapes; `np.testing.*` for arrays; exceptions via
  `pytest.raises(Error, match=…)`, warnings via `pytest.warns`.
- **Parallel-safe** (`pytest-xdist`): write outputs only under `tmp_path`, keep no shared mutable
  state, and never depend on a fixed on-disk path or test order.
- **Git-receipt constraint**: never `chdir` outside the repo (it breaks the receipt's git-SHA
  stamping); CLI subprocess tests run with `cwd`/`PYTHONPATH = _REPO_ROOT`.
- **Determinism**: `np.random.default_rng(<int>)`, never `np.random.seed()`. Constants come from
  `DETECTOR`, never hardcoded.
- **Dtype provenance**: a `TestDtypeProvenance` class per module asserts the §C.4 policy at the
  extension boundaries, internal math, and across a FITS round-trip (via `_dtype_policy.py`); assert
  *precision* (kind + itemsize), **not** the exact dtype object.

#### Profiling

The suite is **"tallest tentpole"** — only the biggest bottlenecks matter, optimization must never
cost scientific accuracy, and "no action needed" is a fine result.

- **Harnesses are not pytest tests** (the `profile_` prefix keeps them uncollected). They mirror the
  test files 1-to-1, run standalone via `make profile*`, take no interactive input, and reference no
  Claude; shared logic lives in `_profiling.py`.
- **Attribute library time to the enclosing KPF method** (`_kpf_attributed`) so a `numpy.partition`
  cost shows against the method that drives it; rank by attributed time and drill into hotspots with
  `line_profiler`.
- **A hotspot** is attributed time both ≥ `HOTSPOT_FRACTION` (20%) **and** ≥ `HOTSPOT_MIN_SECONDS`
  (1 s); no hotspot ⇒ no drill-down. Tune the constants in `_profiling.py`.
- **Reports** print to stdout and write to the gitignored `tests/profiling/reports/`; regenerate them
  when the pipeline's performance profile changes.

---

## D. Documentation

### D.1 Docstrings

- **numpydoc**, with the summary on the same line as the opening `"""` (never its own line); the
  `Parameters`/`Returns`/`Raises`/`Notes` sections follow a blank line, in `name : type` form; a short
  docstring is one line. Wrap code identifiers in **double** backticks (single backticks render as
  italic in RST).
- **Module docstring on every module** — a one-liner for thin files, a multi-paragraph block for
  science modules listing output HDUs (shapes/units) and ending with a `Notes` paper citation
  (`Author (Year) -- reason`). No author tags or dates.
- **Class docstrings document the `__init__` args at the class level** (`Parameters`), not on
  `__init__`; the `Attributes` section is not used — document instance attributes with the
  `# populated by …` comment in `__init__` (§C.2).
- **Document public methods and most private helpers**; give **every exception a caller can trigger**
  its own `Raises` entry (don't omit, understate, or bury it in `Notes`), and document `Returns` as a
  section, not woven into prose. Trivial math primitives may skip docstrings; `Examples` sections are
  not used.
- **Types go in the docstring, not the signature** (§C.3); array shapes/dtypes/units in prose
  (`"WAVE [Å, vacuum]"`).
- **State recurring docstring content once, identically** — the `chip`/`fiber` glossaries; the
  calibration-source override forms (defined once at `ImageProcessing.perform`, referenced elsewhere);
  the `optimize_lsq` `theta = [b, a, mu, sigma]` convention; and the memmap / `to_fits`-shim idioms
  (§B.1).

### D.2 Comments

- **Comments explain *why*, not *what*** — full sentences, capitalized; annotate magic numbers with
  units/meaning (`* 1.48424  # e-/ADU: exposure-meter gain`). If a comment explains *what* a variable
  holds, rename the variable (§C.1).
- **Units in bracket notation** (`[km/s]`, `[Å, vacuum]`, `[e-/ADU]`); state the air/vacuum convention
  wherever wavelengths appear; use astropy `Quantity` for in-code units.
- **Use ASCII `--`, not the em-dash `—`**, in `.py` docstrings/comments (the source is kept ASCII).
- **Code does not cite governing docs** — no "see style guide §…", "charter §…", "EPRV_DATA_STANDARD.md
  §…" in docstrings/comments. State the *reason* a rule exists inline where it helps, but keep the
  citation out of the source; a descriptive term that names a standard ("EPRV-standard ImageHDUs") is
  fine.
- **`TODO` is the only task marker** (`# TODO: …`); no `FIXME`/`XXX`/`HACK`. Legacy-compat choices are
  not annotated in code except in `quicklook`, which documents the v2.12 plots it ports.

---

## E. Open Inconsistencies

Genuine inconsistencies in the codebase with no clear winner yet. Until decided, **match the dominant
variant of the file/area you're editing**, and don't churn unrelated files to "fix" style.

1. **Masters** — the config-resolution block is duplicated 5× (could be a base helper); the `0.2`
   load-failure threshold is an unnamed magic number.
2. **Configs** — the `[DATA_DIRS]` + `[TRACES]` blocks are duplicated verbatim across the science and
   masters configs (no shared-include mechanism).
