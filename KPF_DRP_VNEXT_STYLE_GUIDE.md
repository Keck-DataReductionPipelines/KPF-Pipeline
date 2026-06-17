# KPF-DRP vNext — Style Guide

This guide captures the coding conventions actually in use across the KPF-DRP vNext
codebase, so that new code (whether written by a developer or by Claude) reads like
the code already here. It was derived by surveying every subpackage — standard
modules, masters, quality control, utils, recipes/configs, tests, and documentation.

**Status of these rules.** These are *soft* requirements. They describe the dominant,
prevailing pattern; where the codebase contradicts itself, the recommended variant is
called out explicitly. When following a rule here would conflict with anything in
[`KPF_DRP_VNEXT_CONTEXT.md`](KPF_DRP_VNEXT_CONTEXT.md) (the project charter — the single source of truth for intent, scope, scientific focus, calibration philosophy, and guardrails), **the charter wins.** Style yields to science.

Operational/technical guidance (environment, commands, architecture) lives in
[`CLAUDE.md`](CLAUDE.md); this file covers *how code should look and be organized*.

---

## 1. Naming

| Thing | Convention | Examples |
|---|---|---|
| Modules (files) | `snake_case`, noun phrase = the algorithm stage | `image_assembly.py`, `spectral_extraction.py`, `calibration_association.py` |
| Masters modules | short, single-word, = the product | `bias.py`, `dark.py`, `flat.py`, `wls.py`, `base.py` |
| Classes | `PascalCase`, name = CamelCase of the filename | `ImageAssembly`, `SpectralExtraction`, `RadialVelocity` |
| Acronym classes | keep the acronym capitalized | `WLS` (not `Wls`), `QC` |
| Per-level classes | compact level suffix `L0/L1/L2`, **not** `Level0` | `DiagL0`, `QCL1`, `PlotL0`, `KPF2` |
| Public methods/functions | `snake_case`, verb-led | `count_amplifiers`, `subtract_overscan`, `compute_redshift` |
| Predicates | `is_*`, return `bool` | `is_obs_id`, `is_timestamp` |
| Converters | `<x>_to_<y>` | `air_to_vac`, `utc_to_hst`, `kpf_timestamp_to_eprv_timestamp` |
| Private helpers | leading underscore | `_get_overscan_pixels`, `_resolve_illumination_source` |
| Module constants | `UPPER_SNAKE` | `KECK_LOCATION`, `NORDER`, `SPEED_OF_LIGHT_KMS` |
| Module-private constants | leading underscore | `_RN_KEYS`, `_DEFAULTS`, `_OBS_ID_PATTERN` |
| Variables | `snake_case` | `datecode`, `file_list`, `oscan_srl` |

- **One public class per module** is universal for the pipeline modules. The class name
  is the CamelCase of the file name.
- **Math-heavy locals may be terse single-capital letters** when they mirror a published
  algorithm's notation (e.g. Horne 1986 `D, V, S, F, P, M, W`; Welford `M2`, `N`, `delta`).
  This is sanctioned *only* in numerical code and *only* when the symbols are documented
  in the surrounding docstring/comments. Use descriptive names everywhere else
  (I/O, path handling, orchestration).
- **Module-private constants take a leading underscore** (`_DEFAULTS`, `_RN_KEYS`);
  constants intended for export do not (`NORDER`, `KECK_LOCATION`). Be deliberate about
  which you mean. *(Minor existing inconsistency: a few masters constants like
  `NROW`/`NCOL` are public though module-internal — prefer the underscore for new
  module-private constants.)*
- **FITS keyword names**: ≤ 8 chars, uppercase, no underscores (`NANSCI1`, `ZEROFRAC`,
  `RNINRNG`, `ISGOOD`). Encode the level into the keyword when needed for uniqueness
  (`DATAPRL0`, `L2NANOK`).

---

## 2. Module structure (the "golden skeleton")

Every standard pipeline module follows the same top-to-bottom template. New modules
should match it:

```python
"""KPF <Stage> module.

<Longer modules add a multi-paragraph description, dispatch tables, and a
Notes section citing papers — see radial_velocity.py / barycentric_correction.py.>
"""

import os                                   # 1. stdlib
import warnings

import numpy as np                          # 2. third-party (alphabetical)
import pandas as pd

from kpfpipe import DEFAULTS, DETECTOR       # 3. first-party (absolute only)
from kpfpipe.utils.config import ConfigHandler

_DEFAULTS = {**DEFAULTS, "module_param": ...}   # module constants: _DEFAULTS first
NORDER_GREEN = DETECTOR["norder"]["GREEN"]      # then derived constants / lookup tables


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
        # ... canonical config block (see §4) ...
        self._results = None  # populated by perform()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _helper(self, ...): ...

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------
    def do_step(self, ...): ...

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def perform(self, chips=None):
        ...
        self.l2_obj.receipt_add_entry("stage_name", "PASS")
        return self.l2_obj

    def info(self): ...   # human-readable reporter, always last
```

- **Method order is fixed and marked with 66-dash banner comments**:
  `__init__` → *Private helpers* → *Algorithm steps* → *Public entry point* (`perform`)
  → `info()`.
- The banner is exactly:
  ```python
      # ------------------------------------------------------------------
      # Section name
      # ------------------------------------------------------------------
  ```
  No `=====` or `#####` styles. Sub-labels are allowed
  (`# Private helpers — exposure-meter handling`).
- **Per-level subpackages** (`data_models/`, `quality_control/*/`) use `levelN.py` file
  naming (`level0.py`, `level1.py`, `level2.py`) with shared logic in `base.py`, and
  re-export through `__init__.py` with an explicit `__all__`.

---

## 3. Imports

- **Three groups, blank-line separated, in isort/PEP 8 order**: (1) stdlib,
  (2) third-party, (3) first-party `kpfpipe.*`. Alphabetical within each group.
- **Absolute imports only.** Never relative (`from .base import ...` is not used —
  write `from kpfpipe.modules.masters.base import BaseMasterModule`).
- **Standard aliases**: `import numpy as np`, `import pandas as pd`,
  `import astropy.units as u`, `import matplotlib.pyplot as plt`.
- **Import shared helpers, don't duplicate them.** Detector-geometry helpers
  (`count_amplifiers`, `orient_channels`, `_RN_KEYS`) are owned by `ImageAssembly`;
  other layers import them. Reusable stats/validation live in `kpfpipe/utils/` and are
  imported, never re-implemented (the project's **utils-first** rule).
- **Deferred (in-function) imports are acceptable and used deliberately** in tests and
  occasionally to break import-cost/cycles — when you do it, add a one-line comment
  saying why.
- *Recommended:* run `isort` — a few files (`stats.py`, `validation.py`,
  `barycentric_correction.py`) have within-group ordering drift.

---

## 4. Class design

- **Transform modules are plain standalone classes** — no base class, no mixins, no
  ABCs, no `dataclasses`. They operate *on* the data-model objects, they don't subclass
  them. (The masters and QC/Diag layers *do* share a base class — see §10/§11.)
- **Canonical constructor signature**: `__init__(self, l<N>_obj, config=None)`. The data
  object is the first positional arg, named for its level (`l0_obj`, `l1_obj`, `l2_obj`).
  *(Use `l2_obj`, not `kpf2_obj` — the latter is a lone outlier.)*
- **The config-resolution block is the same everywhere** — copy it verbatim:
  ```python
  if config is None:
      params = {}
  elif isinstance(config, dict):
      params = config
  elif isinstance(config, ConfigHandler):
      params = config.get_params(["DATA_DIRS", "KPFPIPE", "MODULE_<NAME>"])
  else:
      raise TypeError("config must be None, dict, or ConfigHandler")
  for k, v in _DEFAULTS.items():
      setattr(self, k, params.get(k, v))
  ```
  Tunable params become instance attributes via the `_DEFAULTS` setattr loop.
- **`_DEFAULTS` merges the global defaults**: `_DEFAULTS = {**DEFAULTS, "key": ...}`
  (use the `{**DEFAULTS, ...}` spread form, not `dict(DEFAULTS)`).
- **Defaults live in the module, not the config file.** Config (TOML) values are
  *overrides* applied on top of `_DEFAULTS` via `params.get(k, v)`.
- **Declare every lazily-populated attribute in `__init__`, set to `None`/`{}`, with a
  trailing comment naming the method that fills it** — this is a defining house style:
  ```python
  self._results = None       # populated by perform()
  self._line_mask = {}       # line mask, set by _build_line_mask()
  self.ml1_obj = None        # populated by subclass make_master_l1()
  ```
- **Constants come from `kpfpipe.constants` / `DETECTOR` / `detector.toml`**, pulled into
  module-level constants once (`NORDER_GREEN = DETECTOR["norder"]["GREEN"]`). Never
  hardcode order/column counts.
- Use `@staticmethod` for pure functions that touch no instance state.
- **Dispatch-by-name idiom** for pluggable methods:
  `fxn = getattr(self, f"_{method}_extraction")` wrapped in
  `try/except AttributeError → raise AttributeError("Unsupported ...")`.

---

## 5. Function & method design

- **Argument convention — "`None` means use config".** Domain identifiers (`chip`,
  `fiber`, `order`) are positional; every tunable is a keyword arg defaulting to `None`
  and resolved at the top of the method body:
  ```python
  def perform(self, chips=None):
      if chips is None:
          chips = self.chips
  ```
  This is the single most important and most universal convention in the codebase.
- **Keyword-only args (`*`) for optional/tunable parameters** of public helpers:
  ```python
  def build_filepath(obs_id, level, *, data_root=None, master=None): ...
  def _box_extraction(D, V, *, S=None, M=None, W=None): ...
  ```
  Required positionals first; everything tunable after the `*`.
- **String-enum mode parameters** (e.g. `method`, `response`, `imtype`) are validated
  against an explicit allowed set, raising `ValueError` that names the valid options.
- **Return patterns**:
  - A transform's `perform()` returns the next-level data object after mutating
    headers and calling `receipt_add_entry("<module>", "PASS")`.
  - Step methods that mutate `*_obj.data` in place return `None` and document
    *"Modifies … in-place"*.
  - Helpers return a single value or a fixed tuple; validators return `None`.
- **Pure-function discipline in utils/numerical code**: never mutate inputs — `.copy()`
  (or `np.asarray(x).copy()`) before modifying.
- **Length**: helpers ~20–60 lines, `perform()` ~30–70; heavy numerical methods may run
  longer. `perform()` reads as a linear pipeline of step-method calls, typically inside
  `for chip in chips:`.

### Type hints

- **Current reality: type hints are essentially absent.** Types are documented in NumPy
  docstrings instead (the lone exception is `utils/config.py`, which uses modern
  `str | Path` / `dict | None` unions).
- **This is a known gap, not a deliberate prohibition** — `mypy kpfpipe/` is listed in
  CLAUDE.md, implying hints are aspirational. *Recommended direction:* add PEP 484 hints
  to new public functions using the modern `X | None` union syntax. Pending a project
  decision, at minimum keep documenting types in the docstring. **Flag for the team to
  adjudicate.**

---

## 6. Error handling, validation & logging

- **No `logging` module anywhere.** Don't introduce one without a project decision.
  - Human-facing status → `print()`, confined to `info()` reporters and recipe banners.
  - In-pipeline recoverable/degraded conditions → `warnings.warn(...)`, gated behind a
    `verbose` flag: `if verbose: warnings.warn(...)`.
- **Raise exceptions; do not catch-and-log.** Choose the semantically correct type:
  - `TypeError` — wrong `config` type (every `__init__`).
  - `ValueError` — bad domain value / failed validation (the workhorse).
  - `LookupError`/`KeyError` — missing trace/header.
  - `FileNotFoundError` — missing master/input.
  - `NotImplementedError` — stubs.
  - `RuntimeError` — wrapping a failure from a sub-step (QC/Diag runners re-raise as
    `RuntimeError(f"... {name!r} raised: {e}") from e`).
- **No `assert` statements** for validation — use explicit `if ...: raise`.
- **Validate early, fail fast**: check args at the top of the function before doing work.
- **Error messages are f-strings that state the expectation and show the offending
  value with `!r`**: `raise ValueError(f"data_root must be a non-empty string; got {data_root!r}")`.
- **Narrow your `except`**: `except (FileNotFoundError, IOError, OSError) as e`, never
  bare `except:`. Use `raise ... from e` when re-raising. Broad `except Exception` is
  acceptable only around external I/O that converts to a warning-and-skip.
- **Predicate/validator split**: internal `_validate_*` raise; public `is_*` wrap them in
  `try/except` and return `bool` (never raise from the boolean API).
- **Configurable severity**: validators that serve both soft and hard contexts take a
  `response="warn"|"error"|"silent"` channel (see `utils/validation.py`).

---

## 7. NumPy & numerical idioms

- **Vectorize**; use explicit broadcasting (`[:, None]`, `[None, :]`) with named
  temporaries. Loops only for inherently sequential work.
- **Be NaN-aware by default**: `np.nanmedian`, `np.nanstd`, `np.nanmean`; fill missing
  data with `np.full(..., np.nan, dtype=np.float32)`.
- **Be explicit about dtype, and document why when it's not obvious**:
  - Science/storage arrays → `np.float32`.
  - Wavelength solutions, physical constants, fit accumulators → `np.float64`
    (precision/stability); cast with `np.asarray(..., dtype=np.float64)`.
  - Cast kernels/weights to the input `dtype` to stop scipy promoting float32→float64.
- **Prefer robust statistics**: median + MAD (`astropy.stats.mad_std(..., ignore_nan=True)`)
  over mean/std for outlier work; guard divisions with a small `eps` (`1e-12`) or
  `np.maximum(N, 1)`.
- **Pre-zero then fill valid pixels** rather than divide-then-clean; use the `where=`
  kwarg of `np.sum` et al. for masked reductions.
- **Views vs copies are deliberate**: slicing yields views on purpose (consistent with
  the data-model "view not copy" philosophy); `.copy()` when you must mutate.
- **Delegate shared numerics to `kpfpipe/utils/`** (`flag_outliers`, `optimize_lsq`,
  `interpolate_bad_pixels`, `compute_redshift`, `strictly_increasing`). `scipy` is the
  numerical backend (`least_squares` with an analytic Jacobian, `ndimage`,
  `interpolate`); log-space parameterization for positive-definite fit params.

---

## 8. Formatting

- **`black` with default settings is the standard**: 88-char target line length,
  double-quote normalization. `black==26.1.0`, `isort==7.0.0`, `flake8==7.3.0` are pinned
  dev deps; there is **no tool config override**, so black's defaults are canonical.
- **Quote style → double quotes** (black default). *This is the single most widespread
  inconsistency in the repo*: roughly half the files use single quotes. New code should
  use double quotes; run `black .` on files you touch to converge them. **Don't** hand-fix
  quotes across files you aren't otherwise editing.
- **f-strings are the only interpolation style.** No `%` or `.format()` (except inside
  `datetime.strftime` codes, and deliberate numeric formatting like `format(x, "g")`).
  Use `{x!r}` for repr, `{x:05d}`/`{hh:02d}` for zero-padding.
- **Let black own alignment** — don't hand-align assignments or dict values with extra
  spaces; black will collapse them.
- 4-space indent; two blank lines between top-level defs, one between methods.
- **Run black before committing.** Several older/larger files predate a uniform black
  pass (110–115-char lines, trailing whitespace, manual alignment survive) — converge
  them opportunistically when editing.

---

## 9. Quality-control layers (Diagnostics / QC / Quicklook)

These three read-only layers under `kpfpipe/quality_control/` share conventions that new
QC code must follow:

- **Read-only discipline.** Diagnostics and QC write **only** to `headers["PRIMARY"]`,
  never to `data`. Quicklook writes only PNGs. When a helper would mutate `l0.data`,
  operate on a `deepcopy` to protect the caller's object.
- **Method-attribute registration + MRO-walk discovery.** Tag a method by assigning an
  attribute immediately after its `def` — there are no decorators:
  ```python
  def nan_counts(self): ...
  nan_counts._diag_name = "nan_counts"          # Diagnostics

  def data_l0_red_green(self): ...
  data_l0_red_green._qc_key = "DATAPRL0"        # QC
  data_l0_red_green._qc_comment = "QC: ..."
  ```
  The base class's `_iter_*` generator walks `type(self).__mro__`, collects callables
  carrying the tag, and dedupes overrides via a `seen` set (subclass beats base).
- **Runners reset `self.results = {}` at entry** (determinism) and wrap each method call
  in `try/except` re-raising as `RuntimeError` — loud failure, no silent suppression.
- **Header writes are always `(value, comment)` 2-tuples.** QC writes integer `0/1` plus
  an `ISGOOD` aggregate. Round floats before writing
  (`round(float(x), 6)`), and cast numpy scalars to Python `int`/`float`.
- **QC comments are namespaced `"QC: ..."`**; Diagnostics comments are bare descriptive
  phrases.
- **Quicklook plotting**: pyplot state-machine API (`plt.figure(figsize=..., tight_layout=True)`);
  `cmap="viridis"`, `origin="lower"`, vmin/vmax from percentiles; templated titles
  `f"L{N} - {Chip} CCD: {obs_id} - {name}"`; a UTC `KPF QLP: … UT` timestamp annotation;
  save as `f"{obs_id}_L{N}_{plotname}_{chip}_zoomable.png"` with `plt.close(fig)` after
  each. Unimplemented plots are stubbed with a docstring citing the v2.12 source and
  `raise NotImplementedError(...)`.
- *Recommended (existing inconsistencies):* Quicklook lacks a shared base class while
  Diag/QC have one, and uses a `_PLOT_METHODS` tuple instead of method-tags — unifying it
  would match the other two layers. Plot DPI (150 vs 600) and axis fontsizes (14 vs 18)
  drift between L0 and L1; pick one. The `(value, comment)`-unwrap helper is duplicated
  ~4× — factor a single shared helper.

---

## 10. Masters subpackage (`modules/masters/`)

The masters layer is a **batch/stack builder** and diverges from transform modules in
documented, intentional ways — follow *its* conventions when adding masters code:

- **Constructed from a file list, not a data object**: `Bias(l0_file_list, config)`.
- **Entry points are `make_master_l1(...)` / `make_master_l2(...)`**, not `perform()`.
- **Shared base class `BaseMasterModule`** holds all heavy lifting (frame loading,
  stacking, Welford streaming, save). Subclasses (`Bias`, `Dark`, `Flat`, `WLS`) are thin:
  an `__init__` that selects the config section, the `make_master_*` entry point, and an
  `info()`. `_DEFAULTS` extends the base: `{**BaseMasterModule._DEFAULTS, ...}`.
- **Two stacking backends switched by frame count**: in-memory datacube below
  `nframe_stream`, streaming **Welford** accumulation above it (to bound memory). Welford
  runs are delimited by explicit `# Welford algorithm accumulation begins/ends` comment
  fences; accumulators are conceptually float64-for-stability, stored master is float32.
- **Sigma-clipping** via `kpfpipe.utils.stats.flag_outliers(arr, sigma, axis=0)`; robust
  WLS-coefficient combination via `mad_std` + median.
- **`info()` pretty-printer** (ASCII-table summary) is expected on every subclass —
  the `Dark`/`Flat` stubs that lack one are incomplete.
- *Recommended (existing inconsistencies):* the 5×-duplicated config block should be
  factored into a base helper; the repeated `0.2` load-failure threshold should be a
  named constant (`MAX_LOAD_FAILURE_FRAC`) and a stray `f""`-with-no-interpolation fixed;
  keyword-arg call sites with spaces around `=` should be black-normalized; `ord` shadows
  a builtin in `calculate_wls_coeffs`.

---

## 11. Recipes & configuration

- **Recipes are plain Python modules** (not a DSL/`.recipe` file), one `def main(config, args)`
  entry point, no top-level execution. The CLI driver (`tools/cli.py`) imports and calls
  `main`.
- **Module invocation idiom = instantiate then call**, with the variable named
  `snake_case` after the class:
  ```python
  image_assembly = ImageAssembly(l0, config)
  l1 = image_assembly.perform()
  ```
  - Transform modules → `.perform()`; QC/Diag/Quicklook → `.run()` (or `.run("all")`);
    masters → `.make_master_l1/l2()`. The constructor takes the **whole `ConfigHandler`**,
    not pre-extracted params — each module pulls its own section internally.
- **All path construction goes through utils helpers** (`build_filepath`, `build_qlp_dir`,
  `build_l0_file_lists`, `get_obs_id`), never string concatenation. Level passed as a
  literal `"L0"/"L1"/"L2"/"L4"`. `os.makedirs(os.path.dirname(path), exist_ok=True)`
  before every `.to_fits()`.
- **Arg validation**: guard at the top, `raise SystemExit("Error: --obs_id is required …")`
  with an example.
- **Recipe comments** are terse, lowercase, imperative, and explain the *why* (science
  rationale), placed above the stage; transforms annotated with `-->`. Future/disabled
  stages are kept as commented-out blocks mirroring the live pattern, labelled
  `(not yet implemented)`.

### Config files

- **TOML, loaded with stdlib `tomllib`.** One config per recipe, sharing the basename
  (`recipes/kpf_drp_science.py` ↔ `configs/kpf_drp_science.toml`).
- **Sections are `UPPER_SNAKE` in brackets.** Two fixed globals `[DATA_DIRS]`,
  `[KPFPIPE]`; per-module sections are **`[MODULE_<NAME>]`** matching the class
  (`[MODULE_RADIAL_VELOCITY]` ↔ `RadialVelocity`). Every module calls
  `config.get_params(["DATA_DIRS", "KPFPIPE", "MODULE_<NAME>"])`.
- **Key casing**: `lower_snake_case` everywhere except `[DATA_DIRS]` (which uses
  env-var-like `UPPER_SNAKE`). Booleans `true`/`false`; paths double-quoted; lists are
  TOML arrays with explicit `.0` on floats.
- *Recommended (existing inconsistency):* the masters config uses bare `[BIAS]`/`[DARK]`/
  `[FLAT]`/`[WLS]` sections without the `MODULE_` prefix — the dominant convention is the
  prefix; either align masters or document it as a deliberate exception. The
  `[DATA_DIRS]`+`[KPFPIPE]` blocks are duplicated verbatim across configs (drift risk).

### CSV config tables (`data_models/config/`)

- Comma-separated, single header row, no quoting, read with `pandas.read_csv`.
- **Extension manifests** (`L0-extensions.csv`, …): columns
  `HDU,Name,DataType,Required,Description` — integer 0-based HDU index, `UPPER_SNAKE`
  ext name, FITS HDU class, Python-cased `True`/`False`, free-text description. Every CSV
  ends with a `Description` column.
- **Mapping tables**: `aliases.csv` → `KPF,EPRV,Description` (1:1 non-trace aliases only);
  `trace-map.csv` → `Trace,Fiber,Description` (trace/fiber aliases derived at runtime).
- These CSVs are the source of truth for HDU layout and alias registration — keep fiber
  names in sync across `trace-map.csv`, `[KPFPIPE].fibers`, and `detector.toml`.

---

## 12. Tests

- **pytest, class-based.** Group tests in `Test<Subject>` classes; **no bare module-level
  `def test_` functions**. Methods are `test_<behavior>` in `snake_case`; error-path tests
  suffix `_raises`/`_raises_<error>`. Test files are `test_<module>.py` mirroring the
  source; per-level naming follows `data_models/` (`test_quicklook_l0.py`).
- **Section the file with the same 66-dash banner comments** used in source modules.
  Open with a module docstring stating scope and data requirements.
- **Fixtures** (`@pytest.fixture`): module-level when shared, nested in a class when
  scoped to it; named for the object produced (`synthetic_l0_file`, `l2_from_flat`). Use
  `scope="class"` for expensive real-data pipelines (with `tmp_path_factory`, since
  `tmp_path` is function-scoped). Returning `(result, helper_obj)` tuples is common.
- **Test data**: real KPF FITS is **vendored** under `tests/testdata/<LEVEL>/<date>/`,
  referenced via `Path(__file__).parent / "testdata" / ...` assigned to `UPPER_CASE`
  module constants. Two explicit tiers, documented in the module docstring: **synthetic
  in-memory FITS** (astropy) for unit tests, **real `testdata/` FITS** for
  regression/integration tests in their own `Test...RealData`/`Test...Regression` classes.
- **Float tolerances** (use the prevailing pattern for the situation):
  - Analytic recovery → `np.testing.assert_allclose(rtol=1e-5, atol=1e-5)`.
  - FITS round-trips → `assert_array_almost_equal(decimal=4)`.
  - Scalars → `pytest.approx` (bare, or `rel=1e-4`/`abs=...`).
  - Real-data sanity → bare range comparisons (`1.0 < rn < 20.0`).
- **Assertions**: bare `assert` for scalars/shapes/membership/exceptions; `np.testing.*`
  for arrays. Exceptions via `pytest.raises(Error, match="<regex>")` (the `match=` is used
  consistently); warnings via `pytest.warns(...)`; "no warning" via
  `warnings.catch_warnings()` + `simplefilter("error", UserWarning)`. Add an f-string
  assertion message where a failure needs context (loops, subprocess checks).
- **Isolation**: `monkeypatch` is the dominant tool (stub expensive/data-dependent steps);
  `unittest.mock` (`MagicMock`/`patch`) where call objects are needed; hand-rolled stub
  classes for lightweight fakes; `tmp_path`/`tmp_path_factory` for filesystem isolation.
- **Git-receipt / cwd constraint**: in-process tests don't `cd`; CLI tests run a
  subprocess with `cwd=_REPO_ROOT` and `PYTHONPATH=_REPO_ROOT`, where
  `_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`. This is the
  canonical accommodation.
- **Determinism**: seed randomness with `np.random.default_rng(<int>)` (commonly `42`) —
  the modern Generator API. Never `np.random.seed()`. *(A few `test_data_models.py`
  fixtures use unseeded `np.random.random`; prefer seeded `default_rng` for new tests.)*
- **Constants come from `DETECTOR`** (`NORDER_GREEN = DETECTOR["norder"]["GREEN"]`) — never
  hardcode order/column counts.
- *Recommended (existing gaps):* there is **no `conftest.py`**, and synthetic-FITS
  construction is duplicated across many files — a shared factory fixture is the largest
  structural improvement available. Several module docstrings claim
  "skipped if KPF_TESTDATA not set" but no such skip exists (data is vendored) — either
  implement the `@pytest.mark.skipif(... .is_file(), reason=...)` guard (as
  `test_wavelength_calibration.py` does) or drop the misleading docstring.

---

## 13. Docstrings & comments

- **NumPy/numpydoc is the docstring standard.** Opening `"""` on its own line, summary
  on the next line, `Parameters`/`Returns`/`Raises`/`Notes` sections with the
  `name : type` form. Wrap referenced identifiers in backticks. *(A handful of files use
  Google `Args:` — `utils/kpf.py`, `utils/pipeline.py`, `quicklook/level0.py/level1.py`;
  new code should use NumPy.)*
  ```python
  def compute_doppler_factor(v):
      """
      Relativistic Doppler factor f = lambda_obs / lambda_rest …

      Parameters
      ----------
      v : astropy.units.Quantity
          Radial velocity (any velocity unit); positive = receding.

      Returns
      -------
      float or ndarray
          Dimensionless Doppler factor.
      """
  ```
- **Module docstrings on every module.** One-liner for thin files
  (`"""KPF <Stage> module."""`); a rich multi-paragraph block for science modules,
  listing output HDUs with shapes/units and ending with a `Notes` section citing papers
  (`Author (Year) — reason`). No author tags, dates, or `__author__`.
- **Class docstrings document the `__init__` args at the class level** (in a `Parameters`
  section) — *not* on `__init__` itself. The `Attributes` section is **not** used; instead
  document instance attributes with the trailing `# populated by …` comment on the
  assignment in `__init__` (see §4).
- **Document public methods and most private helpers.** Trivial math primitives may skip
  docstrings; short helpers may use a one-line summary, but anything with more than one
  non-obvious argument should get full `Parameters`/`Returns` sections.
- **`Examples` sections are not used.** Worked examples, where helpful, go in the prose of
  the summary (and double as test oracles for pure utils).
- **Types are documented in the docstring, not the signature** (see §5 on the type-hint
  gap). Array shapes/dtypes/units are stated in prose
  (`"(rvdata-standard ImageHDUs, shape (NORDER,))"`, `"WAVE [Å, vacuum]"`).
- **Inline comments explain *why*, not *what*** — full sentences, capitalized. Annotate
  magic numbers with units/meaning inline (`* 1.48424  # e-/ADU: exposure meter gain`).
- **Units use bracket notation** (`[km/s]`, `[Å]`, `[Å, vacuum]`, `e-/ADU`) in
  docstrings/comments — not encoded in variable names (except constant suffixes like
  `SPEED_OF_LIGHT_KMS`). State the **air/vacuum convention** wherever wavelengths appear;
  use astropy `Quantity` for in-code units.
- **`TODO` is the only task marker** (`# TODO: <imperative or open question>`); no
  `FIXME`/`XXX`/`HACK`/`NOTE:`. No issue/ticket linkage is the current norm.
- **Provenance**: legacy v2.12 compatibility choices are flagged with inline `#` comments
  ("Match legacy WLS header convention exactly"); generating scripts are cross-referenced
  in comments ("see scripts/build_rough_wls_from_legacy_wls.py").

---

## 14. Cross-cutting open questions for the team

These are the genuine inconsistencies the survey surfaced where the codebase has no clear
winner or contradicts a stated tool/command. Worth a deliberate project decision:

1. **Quote style** — single vs double split roughly 50/50 across files. Black's default is
   double. *Recommendation:* adopt double, converge with `black .` opportunistically.
2. **Type hints** — absent everywhere except `utils/config.py`, yet `mypy kpfpipe/` is a
   documented command. Decide whether to adopt PEP 484 hints project-wide or formally
   document the docstring-types-only stance and drop the mypy expectation.
3. **Masters config sections** — `[BIAS]` vs the dominant `[MODULE_<NAME>]` prefix.
4. **Quicklook** — no shared base + tuple-based registration, unlike Diag/QC; plus DPI and
   fontsize drift between L0/L1.
5. **Tests** — no `conftest.py`; synthetic-FITS construction duplicated widely; misleading
   "skip if no testdata" docstrings vs vendored data.

Until decided, **match the dominant variant of the file/area you're editing**, and don't
churn unrelated files to "fix" style.
