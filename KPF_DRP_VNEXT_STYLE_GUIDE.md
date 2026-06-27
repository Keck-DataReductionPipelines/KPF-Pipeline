# KPF-DRP vNext — Style Guide

This guide captures the coding conventions actually in use across the KPF-DRP vNext
codebase, so that new code (whether written by a developer or by Claude) reads like
the code already here. It was derived by surveying every subpackage — standard
modules, masters, quality control, utils, recipes/configs, tests, and documentation.

**Status of these rules.** These are *soft* requirements, and this guide sits at the **bottom**
of the project's authority hierarchy:
**1. the WMKO technical requirements ([`WMKO_REQUIREMENTS.md`](WMKO_REQUIREMENTS.md)) →
2. the EPRV data standard ([`EPRV_DATA_STANDARD.md`](EPRV_DATA_STANDARD.md)) →
3. the project charter ([`KPF_DRP_VNEXT_CHARTER.md`](KPF_DRP_VNEXT_CHARTER.md)) → 4. this style
guide.** They describe the dominant, prevailing pattern; where the codebase contradicts itself,
the recommended variant is called out explicitly. When following a rule here would conflict
with any of the three documents above — the WMKO requirements (Keck's binding technical
requirements), the EPRV standard (data-product format), or the charter (intent, scope,
scientific focus, calibration philosophy, guardrails) — **the higher document wins.** Style
yields to science.

This file covers *how code should look and be organized* — not operational or technical
guidance (environment, commands, architecture), which is documented separately.

---

## 1. Naming

- **Code is self-documenting.** Names — variables, functions, methods — should make intent
  legible on their own, so a reader rarely needs a comment to understand *what* a line does.
  This is the primary lever for the §13 rule that comments explain *why*, not *what*: if you
  reach for an inline comment to say what a variable holds, prefer a clearer name instead.
  Balance clarity against brevity — favor intelligible names, but keep sanctioned terse
  notation (documented algorithm symbols; domain-standard short names like `chip`, `order`,
  `oscan`) where it aids rather than hurts readability.

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
| Public constants | `UPPER_SNAKE`; allowed in `data_models/` and the package root, **never in `modules/`** | `REPO_ROOT`, `DEFAULTS`, `DETECTOR`, `L0_EXTENSIONS` |
| Module constants | `UPPER_SNAKE` with a leading underscore — modules export **no** importable constants (one documented exception: `ImageAssembly.RN_KEYS`) | `_DEFAULTS`, `_LEVEL_BY_CAL_TYPE`, `_OBS_ID_PATTERN` |
| Variables | `snake_case` | `datecode`, `file_list`, `oscan_srl` |

- **One public class per module** is universal for the pipeline modules. The class name
  is the CamelCase of the file name.
- **Math-heavy locals may be terse single-capital letters** when they mirror a published
  algorithm's notation (e.g. Horne 1986 `D, V, S, F, P, M, W`).
  This is sanctioned *only* in numerical code and *only* when the symbols are documented
  in the surrounding docstring/comments. Use descriptive names everywhere else
  (I/O, path handling, orchestration).
- **Modules define no public (importable) constants.** Every module-level constant in
  `kpfpipe/modules/` is an implementation detail and takes a leading underscore
  (`_DEFAULTS`, lookup/dispatch tables, internal file paths). Values other code needs are
  *not* re-declared as module constants: pull detector geometry from `DETECTOR`/`detector.toml`
  (already exposed on every instance as `self.norder`, `self.ccd`, …), take physical
  constants from `astropy`, and attach fixed objects (e.g. a site `EarthLocation`) to the
  class as a class attribute. Public `UPPER_SNAKE` constants are fine in `data_models/` and
  the package root, where importing them is intended.
- **One sanctioned exception:** `ImageAssembly.RN_KEYS` (the per-amplifier read-noise
  header-keyword table) is public. `ImageAssembly` is the first module to touch raw L0 and
  owns detector read-noise metadata that QC/Quicklook import rather than re-derive; the
  exception is documented at its definition. Don't add new public module constants on this
  precedent without the same justification.
- **FITS keyword names**: ≤ 8 chars, uppercase, no underscores (`NANSCI1`, `ZEROFRAC`,
  `RNINRNG`, `ISGOOD`). Encode the level into the keyword when needed for uniqueness
  (`DATAPRL0`, `L2NANOK`). **Before inventing a new PRIMARY/extension keyword, grep
  `reference/legacy_data_format.rst` and reuse the legacy spelling/casing wherever the
  science meaning matches** (e.g. `WLSFILE`, `BIASFILE`) — so downstream tools, notebooks,
  and archival workflows keep reading v3 products unchanged. Only coin a new keyword when
  the concept genuinely doesn't exist in the legacy schema.

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

from kpfpipe import DEFAULTS                 # 3. first-party (absolute only)
from kpfpipe.utils.config import ConfigHandler

_DEFAULTS = {**DEFAULTS, "module_param": ...}   # module constants are private (leading _)
_LOOKUP_TABLE = {...}                           # internal maps/paths only — no public constants


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
        self._info = None  # info() summary only

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
        """Populate _info from instance attributes (takes only chips/fibers)."""
        self._info = {...}

    def _set_headers(self, l2_obj):
        """Sole place this module writes headers; reads instance attributes."""
        l2_obj.set_keyword(KEY, self._attr)    # routed to its registry extension

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def perform(self, chips=None):
        ...                                    # populate header-source attributes
        self._set_headers(self.l2_obj)         # consolidates ALL header writes
        self._track_info(chips)                # populates _info, just before the receipt
        self.l2_obj.receipt_add_entry("stage_name", "PASS")
        return self.l2_obj

    def info(self): ...   # human-readable reporter, always last
```

- **Method order is fixed and marked with 66-dash banner comments**:
  `__init__` → *Private helpers* → *Algorithm steps* → *Private helpers - module execution*
  (`_track_info`, then `_set_headers`) → *Public entry point* (`perform`) → `info()`.
- **Header consolidation & `_info`.** A module writes headers in exactly one place — a private
  `_set_headers(obj)` that sources its values from instance attributes (never recomputes,
  never reads another product) and writes each registered keyword via `obj.set_keyword(key, value)`
  (routing + comment come from the registry), called immediately before `receipt_add_entry`. Modules
  that write no header keywords keep an empty helper for uniformity. `_info` (formerly `_results`) is a pruned,
  human-readable summary consumed **only** by `info()` — never the science/header chain, and never
  tests (tests assert on the underlying attributes). It is populated by a private
  `_track_info(self[, chips, fibers])` (no other args — it sources from instance attributes),
  called immediately after `_set_headers` and before the receipt.
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
- **Porting from the legacy DRP**: the v2.12 pipeline is kept at `legacy/KPF-Pipeline/`
  as an algorithm reference. When porting a stage, read the original
  `modules/<stage>/src/alg.py` for the core logic, then rewrite it to the skeleton above —
  do **not** carry over hidden state, database dependencies, or implicit calibration-source
  hierarchy (charter §9 Guardrails, §10 Core Design Principles).

---

## 3. Imports

- **Three groups, blank-line separated, in PEP 8 order**: (1) stdlib,
  (2) third-party, (3) first-party `kpfpipe.*`. Alphabetical within each group. Enforced and
  auto-sorted by Ruff's import rules (`I`; `known-first-party = ["kpfpipe"]`).
- **Absolute imports only.** Never relative (`from .base import ...` is not used —
  write `from kpfpipe.modules.masters.base import BaseMasterModule`).
- **Standard aliases**: `import numpy as np`, `import pandas as pd`,
  `import astropy.units as u`, `import matplotlib.pyplot as plt`.
- **Import shared helpers, don't duplicate them.** Detector-geometry helpers
  (`count_amplifiers`, `orient_channels`) and the public `RN_KEYS` read-noise table are
  owned by `ImageAssembly`; other layers import them. Reusable stats/validation live in
  `kpfpipe/utils/` and are imported, never re-implemented (the project's **utils-first** rule).
- **Deferred (in-function) imports are acceptable and used deliberately** in tests and
  occasionally to break import-cost/cycles — when you do it, add a one-line comment
  saying why.

---

## 4. Class design

- **Transform modules are plain standalone classes** — no base class, no mixins, no
  ABCs, no `dataclasses`. They operate *on* the data-model objects, they don't subclass
  them. (The masters and QC/Diag layers *do* share a base class — see §10/§11.)
- **Canonical constructor signature**: `__init__(self, l<N>_obj, config=None)`. The data
  object is the first positional arg, named for its level (`l0_obj`, `l1_obj`, `l2_obj`).
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
- **Defaults live in the module, not the config file.** Resolution is a three-tier
  override chain, lowest precedence first: `_DEFAULTS` (the in-module default) → config
  (TOML values applied on top via `params.get(k, v)` in the loop above) → a direct keyword
  argument on a method call (overrides both). Config is the production override path;
  direct kwargs are the developer/interactive path (e.g. notebooks), not used in production.
- **Declare every lazily-populated attribute in `__init__`, set to `None`/`{}`.** Add a
  trailing comment naming the filling method (and its shape) **only where that isn't obvious**;
  keep comments minimal otherwise. Conventional attributes (`_info`) stay bare:
  ```python
  self._info = None
  self._ccd_bjd = None   # per-CCD [GREEN, RED] arrays for _set_headers
  self._line_mask = {}   # set by _build_line_mask()
  ```
- **Detector geometry comes from `DETECTOR` (sourced from `detector.toml`), consumed on
  the instance** — every module gets `self.norder` (`{GREEN, RED}` dict), `self.ccd`,
  `self.chips`, `self.fibers` via the `_DEFAULTS` config loop. Do **not** re-declare them as
  module-level constants; use `self.norder["GREEN"]` etc., with a method-local handle for a
  verbose derived value (`norder = self.norder["GREEN"] + self.norder["RED"]`). Never
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
- **Canonical `perform()` signature**:
  ```python
  def perform(self, chips=None, fibers=None, *, <None-kwargs>, <default-value-kwargs>):
  ```
  - **`chips`, `fibers` are the only positional arguments** (each defaulting to `None`),
    in that order. A module that doesn't operate on one of them simply omits it
    (`perform(self, chips=None, *, ...)`); a module that operates on neither has no leading
    positionals. A module whose primary selector is something else keeps that one required
    positional in the same slot (e.g. `CalibrationAssociation.perform(self, cal_types, *, ...)`).
  - **Everything else is keyword-only** — place a bare `*` after the positionals so all
    tunables must be passed by name.
  - **Order the keyword-only args in two groups**, each in sensible domain order: first the
    *configurable* params (backed by `_DEFAULTS` + config, defaulting to `None`, resolving to
    `self.<attr>`), then the *semi-hidden* knobs (a real literal default like `min_npts=9` or
    `verbose=True`, absent from both `_DEFAULTS` and config). A tunable's tier must be legible
    from the signature alone: **`=None` ⇒ configurable**, **literal default ⇒ semi-hidden**. A
    semi-hidden param needing a sequence default uses an *immutable literal*
    (`clip_edge_pixels=(500, 500)`), never a `None`-sentinel + in-body list fallback.
- **The `make_master_*` entry points follow the same shape** (§10), with `l0_file_list`
  as the sole positional in place of `chips`/`fibers`.
- **Parameter ordering applies to *every* method**, not just the public entry points:
  required positionals first, then defaulted parameters in two groups — `None`-defaults
  (configurable) before real-value defaults (semi-hidden). This holds for algorithm step
  methods, private helpers, and standalone utils alike.
- **The `*` (forcing keyword-only) is reserved for public entry points** — `perform()` and
  `make_master_*()`. Private/algorithm methods keep the same parameter *ordering* but do
  **not** need a `*` added; their domain identifiers stay positional. A helper may still use
  `*` where it already aids clarity (existing practice), but don't add one mechanically:
  ```python
  def build_filepath(obs_id, level, *, data_root=None, master=None): ...
  def _box_extraction(D, V, *, S=None, M=None, W=None): ...
  ```
  Required positionals first; everything tunable after the `*`.
- **String-enum mode parameters** (e.g. `method`, `response`, `cal_type`) are validated
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

- **Adopted stance: docstring types only — no PEP 484 annotations.** Document parameter
  and return types in NumPy-style docstrings (`name : str or pathlib.Path`), not in
  signatures. Do not add inline type hints to new code; the codebase carries none.
- **`mypy` is not used.** It still appears as a dev dependency (`pyproject.toml`,
  `environment.yml`), but it is not run or enforced and there are no annotations for it
  to check. Treat those entries as vestigial, not a signal to start annotating.

---

## 6. Error handling, validation & logging

- **No `logging` module anywhere.** Don't introduce one without a project decision.
  - Human-facing status → `print()`, confined to `info()` reporters and recipe banners.
  - In-pipeline recoverable/degraded conditions → `warnings.warn(..., stacklevel=2)`, gated
    behind a `verbose` flag: `if verbose: warnings.warn(..., stacklevel=2)`. The explicit
    `stacklevel` is required (Ruff `B028`) so the warning points at the caller.
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
- **Narrow your `except`**, never bare `except:`. Broad `except Exception` is acceptable
  only around external I/O that converts to a warning-and-skip. **Always parenthesize a
  multi-type clause** — `except (ValueError, TypeError):`, binding or not: the bare PEP 758
  form (`except A, B:`) is valid on 3.14 but Pylance/Pyright can't parse it (§8).
- **Always chain re-raises** (Ruff `B904`): `raise ... from e` to preserve the original
  context, or `raise ... from None` when translating a low-level error (a `KeyError`/
  `AttributeError` from a dict lookup or `getattr` dispatch) into a clearer domain error
  whose message already says what the original would — suppressing the redundant chain.
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
- **Dtype precision is a contract — guard both directions.** Never upscale
  `float32→float64` (memory/throughput regression) nor downscale `float64→float32`
  (precision loss → wrong RVs). The policy — single source of truth, also encoded for
  tests in [`tests/regression/_dtype_policy.py`](tests/regression/_dtype_policy.py):
  - **float32** — L1 `*_CCD`/`*_VAR`, master `*_IMG`/`*_SNR`, L2 `*_FLUX`/`*_VAR`/`*_BLAZE`.
  - **float64** — every `*_WAVE`, `BJD_TDB`, `BARYCORR_KMS`/`_Z`, CCF cubes, and the L4
    RV-table floats (`RV`/`RV_ERR`/`BERV`/`WAVE_START`/`WAVE_END`). `*_WAVE`, `BJD_TDB`,
    `WAVE_START`/`WAVE_END` are **EPRV-mandated 64-bit** (EPRV §2/§3, *born-64 at every
    state* — never rely on RVData's upcast); the rest is KPF precision policy.
  - **bool** in memory / **uint8** (8-bit) on disk — quality masks (`*_MASK`).
  - L0 amps stay native-int or float32 — **never float64**.
  Be explicit at allocation (`np.zeros(..., dtype=...)`, `np.asarray(..., dtype=...)`),
  and cast kernels/weights to the input `dtype` so scipy doesn't promote float32→float64.
  A **deliberate** precision change that produces a higher-precision *result* (a float64
  CCF accumulated from float32 flux) is fine — the result's dtype governs.
- **Prefer robust statistics**: median + MAD (`astropy.stats.mad_std(..., ignore_nan=True)`)
  over mean/std for outlier work; guard divisions with a small `eps` (`1e-12`) or
  `np.maximum(N, 1)`.
- **Pre-zero then fill valid pixels** rather than divide-then-clean; use the `where=`
  kwarg of `np.sum` et al. for masked reductions.
- **Views vs copies are deliberate**: slicing yields views on purpose (consistent with
  the data-model "view not copy" philosophy); `.copy()` when you must mutate.
- **Row/col nomenclature is numpy, not KPF.** All image/spectrum arrays use the numpy axis
  convention throughout: **axis 0** (`row`/`nrow`) = **cross-dispersion** (across orders, flux
  varies rapidly); **axis 1** (`col`/`ncol`) = **dispersion** (along an order, flux varies
  slowly). This is the *transpose* of the KPF/observatory physical convention (where a "row"
  runs along dispersion), so a reader expecting KPF physical directions will misread the code's
  `row`/`col` — but the code is uniform and self-consistent.
  `# Axis convention: axis 0 = cross-dispersion (KPF col); axis 1 = dispersion (KPF row).`
- **Delegate shared numerics to `kpfpipe/utils/`** (`flag_outliers`, `optimize_lsq`,
  `interpolate_bad_pixels`, `compute_redshift`, `strictly_increasing`). `scipy` is the
  numerical backend (`least_squares` with an analytic Jacobian, `ndimage`,
  `interpolate`); log-space parameterization for positive-definite fit params.

---

## 8. Formatting

- **Prefer Ruff's normalization for stylistic nits.** When the formatter has an opinion on a
  purely stylistic point (paren placement, line wrapping, quote style, blank lines), follow
  what `ruff format` produces rather than hand-styling against it — fighting the formatter
  just churns. Deviate only with a strong, documented reason. (Example of such a reason:
  Ruff's `target-version` is pinned to `py313`, one below the 3.14.3 runtime, *so that* the
  formatter keeps the parens on a multi-type `except` instead of emitting PEP 758's bare form
  — which Pylance/Pyright can't parse. The pin makes the formatter agree with the type
  checker; see §6.)
- **`ruff` is the unified formatter + linter** (it replaced black/isort/flake8). The
  formatter is black-compatible: **88-char target line length**, double-quote normalization.
  Config lives in `pyproject.toml` under `[tool.ruff]`; `ruff==0.15.17` and
  `pre-commit==4.6.0` are pinned dev deps. Enforced locally via a pre-commit hook — run
  `pre-commit install` once after setting up the env.
- **Lint ruleset** (`[tool.ruff.lint] select`): `E`/`W` (pycodestyle), `F` (pyflakes),
  `I` (import sorting), `B` (flake8-bugbear), `UP` (pyupgrade). Line length (`E501`) is
  enforced at 88. `__init__.py` is exempt from `F401` (re-exports). Ruff's scope is
  `kpfpipe/`, `tests/`, `recipes/`; `legacy/`, `gjgilbert_notebooks/`, and `scripts/`
  (unimplemented pseudocode stubs) are excluded.
- **Quote style → double quotes** (ruff/black default). The codebase follows the formatter's
  rule (prefer `"`, but keep `'` where switching would add escapes — e.g. `'say "hi"'` stays
  single). Triple-quoted strings use `"""`; f-strings, raw, and byte strings follow the same
  rule. Write new code with double quotes.
- **f-strings are the only interpolation style.** No `%` or `.format()` (except inside
  `datetime.strftime` codes, and deliberate numeric formatting like `format(x, "g")`).
  Use `{x!r}` for repr, `{x:05d}`/`{hh:02d}` for zero-padding.
- **Let the formatter own alignment** — don't hand-align assignments or dict values with
  extra spaces; `ruff format` will collapse them.
- 4-space indent; two blank lines between top-level defs, one between methods.

---

## 9. Quality-control layers (Diagnostics / QC / Quicklook)

These three read-only layers under `kpfpipe/quality_control/` share conventions that new
QC code must follow:

- **Read-only discipline.** Diagnostics and QC write header keywords **only via `set_keyword`**
  (which routes them to QUALITY_CONTROL — see §11 *FITS PRIMARY header conventions*), never to
  `data`. Quicklook writes only PNGs. When a helper would mutate `l0.data`, operate on a `deepcopy`
  to protect the caller's object.
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
- **Writes go through `set_keyword`** (comment sourced from the registry, not the call site).
  QC writes integer `0/1` plus an `ISGOOD` aggregate, then runs `_validate_headers` at the end of
  `run()`. Round floats before writing (`round(float(x), 6)`), and cast numpy scalars to Python
  `int`/`float`. The `_qc_comment`/metric-dict comment is retained in `self.results`, but the FITS
  comment is the registry `Description` — keep the two consistent.
- **QC comments are namespaced `"QC: ..."`**; Diagnostics comments are bare descriptive
  phrases. (Both live in the registry `Description` column, the single source for the FITS comment.)
- **Quicklook plotting**: pyplot state-machine API (`plt.figure(figsize=..., tight_layout=True)`);
  `cmap="viridis"`, `origin="lower"`, vmin/vmax from percentiles; templated titles
  `f"L{N} - {Chip} CCD: {obs_id} - {name}"`; a UTC `KPF QLP: … UT` timestamp annotation;
  save as `f"{obs_id}_L{N}_{plotname}_{chip}_zoomable.png"` with `plt.close(fig)` after
  each. Unimplemented plots are stubbed with a docstring citing the v2.12 source and
  `raise NotImplementedError(...)`.

---

## 10. Masters subpackage (`modules/masters/`)

The masters layer is a **batch/stack builder** and diverges from transform modules in
documented, intentional ways — follow *its* conventions when adding masters code:

- **Constructed from a file list, not a data object**: `Bias(l0_file_list, config)`.
- **Entry points are `make_master_l1(...)` / `make_master_l2(...)`**, not `perform()`. They
  follow the same positional-then-keyword-only shape as `perform()` (§5), with the input
  file list standing in for `chips`/`fibers`:
  ```python
  def make_master_l1(self, l0_file_list=None, *, <None-kwargs>, <default-value-kwargs>):
  ```
  `l0_file_list` is the sole positional (defaulting to `None`, falling back to the list the
  module was constructed with); every tunable is keyword-only after the `*`, ordered
  configurable `None`-kwargs first, then semi-hidden default-value-kwargs (e.g. `verbose=True`).
- **Shared base class `BaseMasterModule`** holds all heavy lifting (frame loading,
  stacking, streaming accumulation, save). Subclasses (`Bias`, `Dark`, `Flat`, `WLS`) are thin:
  an `__init__` that selects the config section, the `make_master_*` entry point, and an
  `info()`. `_DEFAULTS` extends the base: `{**BaseMasterModule._DEFAULTS, ...}`.
- **Two stacking backends switched by frame count**: an in-memory datacube below
  `nframe_stream`, and a single-pass streaming accumulation above it (to bound memory).
  Both sum counts and exposure time per pixel over the surviving (sigma-clipped) frames;
  the master IMG is the exposure-weighted rate `counts_sum / exptime_sum`, stored float32.
- **Sigma-clipping** via `kpfpipe.utils.stats.flag_outliers(arr, sigma, axis=0)`; robust
  WLS-coefficient combination via `mad_std` + median.
- **`info()` pretty-printer** (ASCII-table summary) is expected on every subclass.
- **Public API surface is `make_master*`, `save*`, `stack_frames`, and `info`.**
  `stack_frames` is public so callers can build a stacked frame without the full
  `make_master_*` wrapper. Every other method on `BaseMasterModule` and its
  subclasses — frame loading, calibration, line fitting, coefficient solving, and
  the other algorithm/helper steps — takes a leading underscore. Tests that
  exercise those internals call (and patch) the underscored names directly.

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
- **Product paths go through the `utils/io.py` helpers** (`build_filepath`, `glob_masters`,
  `build_qlp_dir`, `build_l0_file_lists`, `get_obs_id`) — never re-derived by string
  concatenation in modules/recipes. These helpers are the single definition of the on-disk
  layout and filename conventions, built inline within each (no shared layout/filename
  constant or directory helper — that abstraction was deliberately removed as more confusing
  than the duplication). Where a convention is necessarily encoded twice — the masters
  *writer* (`build_filepath`) and *reader* (`glob_masters`) build their paths independently —
  keep them in lock-step with a **drift test** (`test_glob_masters_matches_build_filepath`),
  not a shared helper. Plain `os.path.join` is fine for incidental input-directory assembly
  (e.g. an L0 scan dir). Level passed as a literal `"L0"/"L1"/"L2"/"L4"`;
  `os.makedirs(os.path.dirname(path), exist_ok=True)` before every `.to_fits()`.
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
  `[KPFPIPE]`; per-module sections in the **science** config are **`[MODULE_<NAME>]`**
  matching the class (`[MODULE_RADIAL_VELOCITY]` ↔ `RadialVelocity`), and those modules
  call `config.get_params(["DATA_DIRS", "KPFPIPE", "MODULE_<NAME>"])`.
- **Masters config sections use the bare product name** — `[BIAS]`, `[DARK]`, `[FLAT]`,
  `[WLS]` (no `MODULE_` prefix), matched by `get_params([..., "BIAS"])`. This is a
  deliberate, accepted exception: masters are batch builders keyed by product, not the
  `MODULE_`-prefixed transform stages. Keep it; do not "align" it to the science pattern.
- **Masters that calibrate frames also merge the shared transform-stage sections**
  the per-frame calibration reuses — Dark/Flat/WLS pass
  `get_params([..., "<PRODUCT>", "MODULE_CALIBRATION_ASSOCIATION", "MODULE_IMAGE_PROCESSING"])`
  so the same `bias`/`dark`/`flat` flags and `masters_search_window_days` drive both the
  science path and `_process_frame`. Order matters: `MODULE_IMAGE_PROCESSING` comes **last**
  so its flags win on any key collision. Bias (no per-frame calibration) omits both.
- **Key casing**: `lower_snake_case` everywhere except `[DATA_DIRS]` (which uses
  env-var-like `UPPER_SNAKE`). Booleans `true`/`false`; paths double-quoted; lists are
  TOML arrays with explicit `.0` on floats.

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
- **`L{0,1,2,4}-headers.csv`** register every KPF-pipeline keyword and its home extension,
  split by the level that first writes the keyword (the combined set drives the `set_keyword`
  routing map and the qc_booleans validator). Columns are
  `Keyword,Description,Extension,DataType,PopulatedBy`. The **`Extension`** column is the keyword's
  home header (`PRIMARY`, `QUALITY_CONTROL`, `RECEIPT`, `BJD_TDB`, `BARYCORR_KMS`, `BARYCORR_Z`,
  `RV1`–`RV5`) — `set_keyword` writes there, and `Description` becomes the FITS comment, so a
  keyword's home and comment are defined **once, in the registry**. `DataType` is one of
  `str`/`int`/`float`. Logical flags are stored as `int` 0/1, never Python booleans: QC keys carry a
  `QC: …` description, and every other (T/F) flag appends `(T/F)` to its description. Each `Keyword`
  is an explicit FITS keyword of **≤8 characters** with no wildcards — enumerate every member of
  a family on its own row (e.g. `RNGREEN1`-`RNGREEN4`, `CCD1RV`/`CCD2RV`), never a `?`/`*` stand-in.

### FITS PRIMARY header conventions

The WMKO-native → EPRV-standard conversion happens **only** in `KPF0.to_kpf1`
(`data_models/level0.py`). **Every extension header is an `astropy.io.fits.Header`** — the KPF data models normalize
all headers to `fits.Header` (they override `create_extension`; see `data_models/base.py`),
so there is no value-vs-`(value, comment)` ambiguity and no separate header parser.
`KPF0` owns the WMKO→EPRV conversion (`data_models/level0.py`). The unified registry table
(our `L*-headers.csv` ∪ the EPRV keyword defs) and its derived routing/validation lookups are owned by
the `KeywordRegistry` class in `data_models/keyword_registry.py` — one module singleton
`keyword_registry`, imported only by `base.py`, which surfaces it as the `KPFDataModel.keyword_registry`
class attribute (and uses `.routing` in `set_keyword`); the qc_booleans validator reads `.allowed`/
`.required` off `kpf_obj.keyword_registry`. What this means when writing code:

- **Reading a header value**: use `header.get(key, default)` (or `header[key]`) on the keyword's
  home extension (per the registry `Extension` column — e.g. read `RNGREEN1` from
  `headers["QUALITY_CONTROL"]`, `BIASFILE` from `headers["RECEIPT"]`). A `fits.Header`
  returns the scalar value; the comment lives in `header.comments[key]`. **Never hand-roll
  `value[0] if isinstance(value, tuple)`** — headers are never tuple-valued dicts.
- **Writing a registered KPF-pipeline keyword**: call `obj.set_keyword(key, value)`. It routes the
  keyword to its registry-home extension with the registry `Description` as the comment — never
  hardcode an extension or comment, and never write `headers["PRIMARY"][key] = …` directly for a
  registered keyword. The keyword **must** be in `config/L{0,1,2,4}-headers.csv` first (with its
  `Extension`), or `set_keyword` raises `KeyError` and the qc_booleans validator would reject the
  product. Never write to `INSTRUMENT_HEADER` (immutable snapshot of the L0 PRIMARY as ingested).
- **Writing an unregistered/EPRV-conversion card** (the WMKO→EPRV mapping, provenance stamping):
  the conversion sites in `KPF0` assign `header[key] = (value, comment)` directly; outside those,
  prefer `set_keyword`.
- **Conversion**: call `KPF0.wmko_to_eprv` / `KPF0.build_instrument_header`; don't re-implement the
  WMKO→EPRV mapping.
- **Reading a raw instrument keyword** (`ELAPSED`, `MJD-OBS`, `DATE-OBS`, `GAIAID`, `SCI-OBJ`,
  `TARGTEFF`, …): read it from `headers["INSTRUMENT_HEADER"]` (via `.get`), never from
  PRIMARY. No silent fallback — let a missing key raise.
- Use EPRV keyword *names* on PRIMARY (e.g. `EXPTIME`, not `ELAPSED`; `OBSTYPE`, not `IMTYPE`).

---

## 12. Tests

- **pytest, class-based.** Group tests in `Test<Subject>` classes; **no bare module-level
  `def test_` functions**. Methods are `test_<behavior>` in `snake_case`; error-path tests
  suffix `_raises`/`_raises_<error>`. Test files are `test_<module>.py` mirroring the
  source; per-level naming follows `data_models/` (`test_quicklook_l0.py`).
- **Section the file with the same 66-dash banner comments** used in source modules.
  Open with a module docstring stating scope and data requirements.
- **Fixtures** (`@pytest.fixture`): named for the object produced (`synthetic_l0_file`,
  `l2_from_flat`). Fixtures used by **more than one file live in `tests/conftest.py`**
  (e.g. `synthetic_l0_file`/`synthetic_l1_file` and the seeded `image_hdu` builder); keep
  single-consumer fixtures local. Use `scope="class"` for expensive real-data pipelines
  (with `tmp_path_factory`, since `tmp_path` is function-scoped). Returning
  `(result, helper_obj)` tuples is common.
- **Shared non-fixture helpers** go in an underscore-prefixed module that pytest does not
  collect (`tests/regression/_masters.py`), imported relatively (`from ._masters import ...`) — do not
  duplicate a builder across files or hang it off `conftest.py` (which is for fixtures/hooks).
- **Profiling harnesses** (`tests/profiling/profile_<module>.py`, `tests/profiling/profile_*_recipe.py`, and
  the shared `tests/profiling/_profiling.py`) are *not* pytest tests — the `profile_` prefix keeps them out
  of collection so `make test` stays fast. They **mirror the test files 1-to-1**
  (`test_<x>.py` ↔ `profile_<x>.py`). They are standalone scripts run via `make profile*`,
  must run with **no interactive input**, and must contain **no references to Claude**.
  New profiling logic belongs in `tests/profiling/_profiling.py`, not duplicated per file; each
  `profile_<module>.py` is a thin `setup`/`call` wrapper over `run_profile`.
- **Test data**: real KPF FITS lives under `tests/testdata/<LEVEL>/<date>/`,
  referenced via `Path(__file__).parent / "testdata" / ...` assigned to `UPPER_CASE`
  module constants. Two explicit tiers, documented in the module docstring: **synthetic
  in-memory FITS** (astropy) for unit tests, **real `testdata/` FITS** for
  regression/integration tests in their own `Test...RealData`/`Test...Regression` classes.
  `tests/testdata/` is **intentionally gitignored** (large FITS); the few developers copy
  the needed files locally and coordinate out-of-band. **Never commit anything under it,
  and don't hunt for or build a fixture-generation script — there isn't one, by design.**
  If an integration test needs a missing/stale testdata file, regenerate it **locally** and
  note in the response that the shared copy needs the same update.
- **Markers** (registered in `conftest.py`): mark integration / heavy-compute classes
  `@pytest.mark.slow`, and truth-frame-gated classes `@pytest.mark.requires_testdata`
  (auto-skipped when `tests/testdata/` is absent). The fast pre-commit subset is
  `-m "not slow"`; *when* to run the subset vs the full suite is run-policy, out of scope
  for this style guide.
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
- **Parallel-safe**: the suite runs under `pytest-xdist` (`-n auto --dist loadscope`), so
  every test must be parallel-safe — write outputs only under `tmp_path`, keep no shared
  mutable module/global state, and never depend on a fixed on-disk path or test order.
- **Git-receipt / cwd constraint**: in-process tests don't `cd` — **never `chdir` outside
  the repo**, which breaks the receipt's git-SHA provenance stamping. CLI tests run a
  subprocess with `cwd=_REPO_ROOT` and `PYTHONPATH=_REPO_ROOT`, where
  `_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`. This is the
  canonical accommodation.
- **Determinism**: seed randomness with `np.random.default_rng(<int>)` (commonly `42`) —
  the modern Generator API. Never `np.random.seed()`.
- **Constants come from `DETECTOR`** (`NORDER_GREEN = DETECTOR["norder"]["GREEN"]`) — never
  hardcode order/column counts.
- **Dtype provenance**: each module test file has a `TestDtypeProvenance` class asserting
  the §7 float32/float64/uint8/bool policy at the extension boundaries, the internal
  math-bearing functions (typed-input → output dtype), and across a FITS round-trip, using
  the shared rubric [`tests/regression/_dtype_policy.py`](tests/regression/_dtype_policy.py). Assert *precision*
  (kind + itemsize via `assert_dtype`), **not** the exact dtype object — FITS round-trips
  to big-endian, so `>f4` is still float32.

---

## 13. Docstrings & comments

- **NumPy/numpydoc is the docstring standard.** Opening `"""` on its own line, summary
  on the next line, `Parameters`/`Returns`/`Raises`/`Notes` sections with the
  `name : type` form. Wrap referenced identifiers in backticks. The codebase is uniformly
  numpydoc — no Google `Args:` style remains.
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
- **Types are documented in the docstring, not the signature** (see §5). Array
  shapes/dtypes/units are stated in prose
  (`"(rvdata-standard ImageHDUs, shape (NORDER,))"`, `"WAVE [Å, vacuum]"`).
- **Inline comments explain *why*, not *what*** — full sentences, capitalized. Annotate
  magic numbers with units/meaning inline (`* 1.48424  # e-/ADU: exposure meter gain`). If a
  comment is needed to explain *what* a variable holds, rename the variable instead (§1, code
  is self-documenting).
- **Units use bracket notation** (`[km/s]`, `[Å]`, `[Å, vacuum]`, `[e-/ADU]`) in
  docstrings/comments — not encoded in variable names (except unit-suffixed names like
  `..._KMS`). State the **air/vacuum convention** wherever wavelengths appear;
  use astropy `Quantity` for in-code units.
- **`TODO` is the only task marker** (`# TODO: <imperative or open question>`); no
  `FIXME`/`XXX`/`HACK`/`NOTE:`. No issue/ticket linkage is the current norm.
- **Provenance**: legacy v2.12 compatibility choices are generally **not** annotated in
  code — the `quicklook` module is the one exception (it documents the v2.12 plots it
  ports). Generating scripts may be cross-referenced where it aids reproducibility
  (`"see scripts/build_rough_wls_from_legacy_wls.py"`).

---

### Open Inconsistencies

Genuine inconsistencies in the codebase with no clear winner yet. Until decided, **match
the dominant variant of the file/area you're editing**, and don't churn unrelated files to
"fix" style.

1. **Quicklook** — no shared base class + tuple-based registration, unlike Diag/QC; DPI
   (150 vs 600) and axis-fontsize (14 vs 18) drift between L0 and L1.
2. **Masters** — the config-resolution block is duplicated 5× (could be a base helper); the
   `0.2` load-failure threshold is an unnamed magic number.
3. **Configs** — the `[DATA_DIRS]` + `[KPFPIPE]` blocks are duplicated verbatim across the
   science and masters configs (no shared-include mechanism).
