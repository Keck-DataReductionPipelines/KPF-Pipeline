"""Shared profiling harness for the KPF-DRP vNext "tallest tentpole" suite.

This is **not** a test module (no ``test_`` prefix) and pytest does not collect
it. It is imported by the standalone ``tests/profiling/profile_*.py``
harnesses, run on demand via ``make profile*`` (see the ``## Profiling``
section of CLAUDE.md). It runs fully standalone with no interactive input.

Strategy
--------
We only care about the most critical bottlenecks ("tentpoles"). Each harness:

1. Runs the target under :mod:`cProfile` (pass 1) and charges every function's own
   time to the nearest enclosing **KPF method** (attribution), so the ranking
   points at the method to optimize rather than the library leaf it bottoms out in.
2. Flags the **hotspots** — KPF methods with >= :data:`HOTSPOT_FRACTION` attributed
   time AND >= :data:`HOTSPOT_MIN_SECONDS`. "No hotspot" is a fine outcome.
3. Drills into each hotspot method line-by-line with :mod:`line_profiler` (pass 2)
   to show where inside it the time goes.
4. Emits a human-readable report to stdout *and* to a gitignored Markdown file
   under :data:`REPORTS_DIR`, with a generic, rule-based optimization note per
   hotspot.

Profiling is meaningful only at realistic array sizes, so the harnesses run on
the real (gitignored) frames in ``tests/testdata`` and skip cleanly when those
frames are absent (mirrors the ``requires_testdata`` test pattern).
"""

import ast
import cProfile
import functools
import inspect
import io
import os
import pstats
import tempfile
import textwrap
from datetime import UTC, datetime
from pathlib import Path

from line_profiler import LineProfiler

# --- paths and sample frames -----------------------------------------------

# This module lives in tests/profiling/; testdata and configs are anchored
# relative to it (tests/testdata and the repo-root configs/, respectively).
TESTDATA_DIR = Path(__file__).parent.parent / "testdata"
REPORTS_DIR = Path(__file__).parent / "reports"
CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"

# Canonical sample frames present in tests/testdata/{L0,masters}/20240405.
SCIENCE_OBS_ID = "KP.20240405.40113.57"
MASTERS_DATECODE = "20240405"

# --- thresholds (fractions are of the total own-time budget) ----------------

# A function is a "hotspot" (a tentpole to drill into, and a Recommended action)
# when it clears BOTH of these — a dominant share AND a non-trivial absolute cost.
HOTSPOT_FRACTION = 0.20  # >= this share of own time AND ...
HOTSPOT_MIN_SECONDS = 1.0  # ... >= this many seconds of own time -> a hotspot
TOP_FUNCTION_MIN_FRACTION = 0.02  # >= this share of own time -> listed in the ranking

# --- recipe-report classification -------------------------------------------
# Network time is quarantined from the recipe reports: it is a nondeterministic
# IERS download during barycentric correction, so it would make run-to-run
# comparison meaningless. Matched as substrings against each row's "<file>
# <func>" string.
NETWORK_MARKERS = (
    "_ssl",
    "SSLSocket",
    "ssl.py",
    "socket",
    "urllib",
    "http",
    "do_handshake",
    "certifi",
)

# The masters I/O-vs-compute split measures disk I/O at the data-model
# serialization boundary rather than by low-level primitives: every read is a
# `from_fits` call (in base._load_frame and base._load_calibration) and every
# write a `to_fits` call (in base.save_master). Their cumulative time captures the
# full read/write + parse cost (a leaf-marker heuristic missed ~75% of it — most
# of from_fits is astropy/numpy array construction, not a marked primitive) while
# excluding the compute they sit next to (ImageAssembly runs *after* from_fits
# returns in _load_frame, so it stays in "compute"). See _io_compute_split.
SERIALIZATION_NAMES = frozenset({"from_fits", "to_fits"})

# Labels for the recipe "High-level summary" (cumulative wall-clock per stage,
# a clean partition of the recipe runtime; see _recipe_summary).
IO_SUMMARY_LABEL = "I/O (data-product read/write)"
QUICKLOOK_LABEL = "quicklook (all levels)"
QC_LABEL = "QC + diagnostics (all levels)"
OVERHEAD_LABEL = "(orchestration / overhead)"


# ---------------------------------------------------------------------------
# Data access: skip-if-absent + config builders
# ---------------------------------------------------------------------------


def require_testdata():
    """Exit 0 with a clear message when the real frames are absent.

    Treated as "skipped", never a failure — same contract as the
    ``requires_testdata`` pytest marker in ``tests/conftest.py``.
    """
    if not TESTDATA_DIR.exists():
        print(
            f"[skip] {TESTDATA_DIR} not present — profiling needs real frames; "
            "nothing to do."
        )
        raise SystemExit(0)


def _config(name, output_dir):
    from kpfpipe.utils.config import ConfigHandler

    return ConfigHandler(
        str(CONFIGS_DIR / name),
        overrides={
            "DATA_DIRS": {
                "KPF_DATA_INPUT": str(TESTDATA_DIR),
                "KPF_MASTERS_OUTPUT": str(TESTDATA_DIR),
                "KPF_SCIENCE_OUTPUT": str(output_dir),
            }
        },
    )


def science_config(output_dir=None):
    """ConfigHandler for the science recipe, reading real masters from testdata."""
    output_dir = output_dir or tempfile.mkdtemp(prefix="kpf_profile_sci_")
    return _config("kpf_drp_science.toml", output_dir)


def masters_config(output_dir=None):
    """ConfigHandler for the masters recipe, writing masters to a temp dir."""
    output_dir = output_dir or tempfile.mkdtemp(prefix="kpf_profile_mas_")
    from kpfpipe.utils.config import ConfigHandler

    # Masters recipe writes (and re-reads, for dark's bias association) under
    # KPF_MASTERS_OUTPUT, so that must be the writable temp dir, not testdata.
    return ConfigHandler(
        str(CONFIGS_DIR / "kpf_drp_masters.toml"),
        overrides={
            "DATA_DIRS": {
                "KPF_DATA_INPUT": str(TESTDATA_DIR),
                "KPF_MASTERS_OUTPUT": str(output_dir),
            }
        },
    )


# ---------------------------------------------------------------------------
# Intermediate-product builders (the science L0 -> L2 chain)
#
# Each rebuilds from scratch so a harness can call it twice (the cProfile pass
# and the line_profiler pass each get a fresh, unmutated input). Construction is
# kept OUT of the profiled call so each per-module report isolates that module.
# ---------------------------------------------------------------------------


def load_l0(config):
    from kpfpipe.data_models.level0 import KPF0
    from kpfpipe.utils.io import kpf_filepath

    return KPF0.from_fits(
        kpf_filepath(SCIENCE_OBS_ID, "L0", data_root=str(TESTDATA_DIR))
    )


def assemble_l1(config):
    from kpfpipe.modules.image_assembly import ImageAssembly

    return ImageAssembly(load_l0(config), config).perform()


def process_l1(config):
    from kpfpipe.modules.calibration_association import CalibrationAssociation
    from kpfpipe.modules.image_processing import ImageProcessing

    l1 = assemble_l1(config)
    l1 = CalibrationAssociation(l1, config).perform(["bias", "dark", "flat", "thar"])
    return ImageProcessing(l1, config).perform()


def extract_l2(config):
    from kpfpipe.modules.spectral_extraction import SpectralExtraction

    return SpectralExtraction(process_l1(config), config).perform()


def wls_l2(config):
    from kpfpipe.modules.wavelength_calibration import WavelengthCalibration

    return WavelengthCalibration(extract_l2(config), config).perform()


def bary_l2(config):
    from kpfpipe.modules.barycentric_correction import BarycentricCorrection

    return BarycentricCorrection(wls_l2(config), config).perform()


# --- masters inputs --------------------------------------------------------

# Widen the time_of_day gap so bias/thar's frames form a single representative
# stack rather than splitting morn/eve. Darks instead use groupby='obs_night'
# (the whole night in one stack), matching recipes/kpf_drp_masters.py.
MASTERS_CLUSTER_GAP_SECONDS = 24 * 3600

# Reading masters (bias/dark/flat/thar) for the dark and WLS harnesses, which
# associate pre-built masters rather than constructing them. The bundled masters
# live under tests/testdata/masters, addressed via KPF_MASTERS_OUTPUT.
MASTERS_CONFIG = {"KPF_MASTERS_OUTPUT": str(TESTDATA_DIR)}


def masters_l0_files(imtype):
    """First real L0 cluster of ``imtype`` ('bias'|'dark'|'thar') from testdata."""
    from kpfpipe.utils.io import FileHandler

    file_handler = FileHandler({"KPF_DATA_INPUT": str(TESTDATA_DIR)})
    file_handler.build_mini_database(MASTERS_DATECODE)
    if imtype == "dark":
        kwargs = {"min_file_count": 3, "groupby": "obs_night"}
    else:
        kwargs = {"cluster_gap_seconds": MASTERS_CLUSTER_GAP_SECONDS}
    return file_handler.build_calibration_stacks(imtype, **kwargs)[0]


# ---------------------------------------------------------------------------
# Profiling core
# ---------------------------------------------------------------------------


def _is_own_code(filename):
    """True for KPF pipeline source (not stdlib / site-packages)."""
    return "/kpfpipe/" in filename and "/site-packages/" not in filename


_KPF_PKG_DIR = None


def _kpf_pkg_dir():
    """Absolute path of the installed ``kpfpipe`` package directory (cached)."""
    global _KPF_PKG_DIR
    if _KPF_PKG_DIR is None:
        import kpfpipe

        _KPF_PKG_DIR = os.path.dirname(os.path.abspath(kpfpipe.__file__))
    return _KPF_PKG_DIR


def _is_kpf_module(filename):
    """True for genuine KPF source under the editable-install package dir.

    Anchored to ``kpfpipe.__file__`` rather than the ``/kpfpipe/`` substring used
    by :func:`_is_own_code`: the conda env is itself named ``kpfpipe``, so its
    stdlib lives under ``.../envs/kpfpipe/lib/python3.14/`` and the substring test
    misclassifies every stdlib module as own code. Used by the recipe reports.
    """
    return filename.startswith(_kpf_pkg_dir() + os.sep)


def _module_label(filename):
    """Human label for a kpfpipe source file, e.g. ``modules/radial_velocity.py``."""
    marker = "/kpfpipe/"
    idx = filename.rfind(marker)
    return filename[idx + len(marker) :] if idx >= 0 else filename


def _rank(stats):
    """Return (rows sorted by own time desc, total budget seconds)."""
    total = stats.total_tt or 1e-12
    rows = []
    for (fn, ln, name), (_cc, nc, tt, ct, _callers) in stats.stats.items():
        rows.append(
            {
                "file": fn,
                "line": ln,
                "name": name,
                "ncalls": nc,
                "tottime": tt,
                "cumtime": ct,
                "tot_frac": tt / total,
                "own": _is_own_code(fn),
            }
        )
    rows.sort(key=lambda r: r["tottime"], reverse=True)
    return rows, total


def _kpf_attributed(stats):
    """Charge each function's own time to the nearest enclosing KPF method.

    cProfile aggregates by function, so a bottleneck surfaces as a library *leaf*
    (e.g. ``numpy.partition``) with no hint of *which* KPF method drives it. To
    locate the bottleneck in our own code, push every non-KPF function's own time
    up the caller graph — weighted by per-caller cumulative time — until it lands
    on a KPF method, which intercepts it. A KPF method therefore keeps its own
    Python time *plus* the library/builtin time it directly drives; nested KPF
    methods each get their own row, so the charge stops at the nearest one.

    Returns ``{kpf_func_key: attributed_seconds}`` (``func_key`` is cProfile's
    ``(filename, lineno, funcname)``).
    """
    raw = stats.stats  # {func: (cc, nc, tt, ct, {caller: (cc, nc, tt, ct)})}
    cache = {}

    def lift(func, stack):
        """Distribution ``{kpf_func: weight}`` of ``func``'s own time onto methods."""
        if _is_kpf_module(func[0]):
            return {func: 1.0}  # a KPF method is its own nearest ancestor
        if func in cache:
            return cache[func]
        if func in stack:  # recursion cycle: break it (this weight escapes)
            return {}
        entry = raw.get(func)
        if entry is None:
            return {}
        callers = entry[4]
        # Weight callers by the cumtime each contributes; fall back to call counts.
        weights = {c: v[3] for c, v in callers.items() if v[3] > 0}
        if not weights:
            weights = {c: v[1] for c, v in callers.items() if v[1] > 0}
        tot = sum(weights.values())
        if tot <= 0:
            return {}  # no caller info -> unattributable (weight escapes)
        dist = {}
        sub_stack = stack | {func}
        for c, wt in weights.items():
            frac = wt / tot
            for m, mw in lift(c, sub_stack).items():
                dist[m] = dist.get(m, 0.0) + mw * frac
        cache[func] = dist
        return dist

    attributed = {}
    empty = frozenset()
    for func, vals in raw.items():
        own = vals[2]  # tottime
        if own <= 0:
            continue
        for m, wt in lift(func, empty).items():
            attributed[m] = attributed.get(m, 0.0) + own * wt
    return attributed


def _kpf_rows(stats):
    """KPF-method rows ranked by attributed time (see :func:`_kpf_attributed`)."""
    total = stats.total_tt or 1e-12
    raw = stats.stats
    rows = []
    for func, secs in _kpf_attributed(stats).items():
        fn, ln, name = func
        _cc, nc, tt, ct, _callers = raw[func]
        rows.append(
            {
                "file": fn,
                "line": ln,
                "name": name,
                "ncalls": nc,
                "own_s": tt,
                "cumtime": ct,
                "attr_s": secs,
                "attr_frac": secs / total,
            }
        )
    rows.sort(key=lambda r: r["attr_s"], reverse=True)
    return rows, total


def _is_hotspot(r):
    """Unified selection rule for the drill-down and Recommended actions.

    A KPF method is a hotspot when its **attributed** time is >= both
    :data:`HOTSPOT_FRACTION` of the budget and :data:`HOTSPOT_MIN_SECONDS`.
    """
    return r["attr_frac"] >= HOTSPOT_FRACTION and r["attr_s"] >= HOTSPOT_MIN_SECONDS


def _flags(rows):
    """Hotspot KPF methods — both the Recommended actions and the drill-down set."""
    return [r for r in rows if _is_hotspot(r)]


def _network_own(rows):
    """Total own time of network calls (SSL/socket), for the separate report line.

    Network is a nondeterministic IERS download during barycentric correction, so
    it is quarantined from the recipe's High-level summary.
    """
    return sum(
        r["tottime"]
        for r in rows
        if any(m in f"{r['file']} {r['name']}" for m in NETWORK_MARKERS)
    )


def _io_compute_split(stats):
    """Whole-run time partitioned into disk I/O vs compute (masters recipe).

    Answers "is the recipe read/write-bound or compute-bound?". I/O is the
    cumulative time of the data-model serialization calls — ``from_fits`` (reads,
    in ``base._load_frame`` / ``base._load_calibration``) and ``to_fits`` (writes,
    in ``base.save_master``) — which is where every disk read/write happens. Only
    the *outermost* serialization frame is counted (a caller that is not itself a
    serialization call), so an overridden or otherwise nested ``from_fits`` is not
    double-counted. Compute is the remaining budget — including the image assembly
    that ``_load_frame`` runs *after* the read. Returns ``io``/``compute`` seconds
    and their ``total`` (the percentage base).
    """
    total = stats.total_tt or 1e-12
    io_s = 0.0
    for (_fn, _ln, name), (_cc, _nc, _tt, ct, callers) in stats.stats.items():
        if name not in SERIALIZATION_NAMES:
            continue
        if not callers:
            io_s += ct  # top-level serialization call (no recorded caller)
            continue
        for (_cfn, _cln, cname), caller_vals in callers.items():
            if cname not in SERIALIZATION_NAMES:
                io_s += caller_vals[3]  # per-caller cumtime; outermost frames only
    io_s = min(io_s, total)
    return {"io": io_s, "compute": total - io_s, "total": total}


def _is_quicklook(filename):
    return _is_kpf_module(filename) and _module_label(filename).startswith(
        "quality_control/quicklook/"
    )


def _plot_class(filename):
    """``quality_control/quicklook/level0.py`` -> ``PlotL0`` (the QLP class)."""
    stem = _module_label(filename).rsplit("/", 1)[-1].removesuffix(".py")  # "level0"
    return "PlotL" + stem.removeprefix("level")


def _quicklook_breakdown(stats):
    """Per-plot wall-clock within the quicklook stage (science recipe).

    ``PlotL0/L1/L2.run`` dispatch to each public plot method via ``getattr``
    (e.g. ``PlotL0.stitched_image``), so each method is a direct child of a
    ``run`` and its cumulative time — including the matplotlib render and PNG
    save — is that plot's cost. Underscore helpers (``_has_chip``) and non-
    quicklook children (``os.makedirs``, ``plt.close``) are skipped. Returns
    ``rows`` (``(label, seconds, fraction)``) and the quicklook ``total`` (sum of
    the ``run`` cumtimes); empty when the recipe runs no quicklook (masters).
    """
    run_keys = [k for k in stats.stats if _is_quicklook(k[0]) and k[2] == "run"]
    if not run_keys:
        return {"rows": [], "total": 0.0}
    total = sum(stats.stats[k][3] for k in run_keys)  # cumtime of each run call

    agg = {}
    for func, (_cc, _nc, _tt, _ct, callers) in stats.stats.items():
        filename, _ln, name = func
        if name.startswith("_") or not _is_quicklook(filename):
            continue
        ct = sum(callers[r][3] for r in run_keys if r in callers)
        if ct > 0:  # a plot method dispatched from run()
            label = f"{_plot_class(filename)}.{name}"
            agg[label] = agg.get(label, 0.0) + ct

    base = total or 1e-12
    rows = sorted(
        ((lbl, s, s / base) for lbl, s in agg.items()),
        key=lambda e: e[1],
        reverse=True,
    )
    overhead = total - sum(agg.values())
    if overhead > 0:
        rows.append(("(dispatch / figure close)", overhead, overhead / base))
    return {"rows": rows, "total": total}


def _stage_name(module_label):
    """``modules/radial_velocity.py`` -> ``radial_velocity``; ``modules/masters/
    bias.py`` -> ``bias`` (the product, per the masters "by product" rule)."""
    return module_label[len("modules/") :].removesuffix(".py").split("/")[-1]


def _summary_bucket(filename, name):
    """Map a recipe-``main`` direct child to a (label, kind) High-level bucket.

    ``from_fits``/``to_fits`` are the data-product reads/writes (I/O); the KPF
    pipeline stages, quicklook, and QC/diagnostics map to their respective rows;
    barycentric correction is quarantined; everything else (path helpers, config,
    glue) is orchestration overhead. ``utils``/``data_models`` are deliberately
    *not* their own rows — their time is absorbed into the calling stage.
    """
    if name in ("from_fits", "to_fits"):
        return IO_SUMMARY_LABEL, "io"
    if _is_kpf_module(filename):
        label = _module_label(filename)
        if label.endswith("barycentric_correction.py"):
            return "barycentric", "barycentric"
        if label.startswith("quality_control/quicklook/"):
            return QUICKLOOK_LABEL, "quicklook"
        if label.startswith(
            ("quality_control/qc_flags/", "quality_control/diagnostics/")
        ):
            return QC_LABEL, "qc"
        if label.startswith("modules/"):
            return _stage_name(label), "stage"
    return OVERHEAD_LABEL, "overhead"


def _find_main_key(stats, call):
    """pstats key ``(file, lineno, "main")`` for the profiled recipe entry."""
    code = call.__code__
    key = (code.co_filename, code.co_firstlineno, code.co_name)
    if key in stats.stats:
        return key
    for k in stats.stats:
        if k[2] == code.co_name and k[0] == code.co_filename:
            return k
    return None


def _recipe_summary(stats, call, network_own):
    """Partition recipe wall-clock among its top-level operations (cumulative).

    Each function records its cumulative time *per caller* (cProfile), so every
    direct child call of the recipe's ``main`` is a disjoint slice of wall-clock.
    Grouping those slices by bucket therefore yields a clean partition that sums
    to the recipe runtime — including, e.g., the masters-read I/O nested inside
    ``calibration_association`` (it lands in that stage, never double-counted).

    Barycentric correction is quarantined (nondeterministic IERS fetch): its
    slice is reported separately and excluded from the denominator. Returns the
    sorted ``rows`` (``(label, seconds, fraction, kind)``), the ``denominator``
    (wall-clock minus barycentric), ``bary_seconds``, and ``network_seconds``.
    """
    main_key = _find_main_key(stats, call)
    buckets = {}
    kinds = {}
    bary_seconds = 0.0
    if main_key is not None:
        for func, (_cc, _nc, _tt, _ct, callers) in stats.stats.items():
            if main_key not in callers:
                continue
            ct = callers[main_key][3]  # cumulative time of this call from main
            label, kind = _summary_bucket(func[0], func[2])
            if kind == "barycentric":
                bary_seconds += ct
                continue
            buckets[label] = buckets.get(label, 0.0) + ct
            kinds[label] = kind
        # main's own time = recipe glue not inside any child call.
        main_own = stats.stats[main_key][2]
        buckets[OVERHEAD_LABEL] = buckets.get(OVERHEAD_LABEL, 0.0) + main_own
        kinds[OVERHEAD_LABEL] = "overhead"

    denominator = sum(buckets.values()) or 1e-12
    rows = sorted(
        ((lbl, s, s / denominator, kinds[lbl]) for lbl, s in buckets.items()),
        key=lambda e: e[1],
        reverse=True,
    )
    return {
        "rows": rows,
        "denominator": denominator,
        "bary_seconds": bary_seconds,
        "network_seconds": network_own,
    }


def _note(r):
    """Generic, rule-based optimization hint for a flagged KPF method."""
    # Most of the cost is in the library/builtin calls this method makes.
    if r["attr_s"] - r["own_s"] >= 0.5 * r["attr_s"]:
        return (
            "cost is mostly in library calls it makes — see the drill-down for the "
            "hot line; cut the call count or batch the array op"
        )
    if r["ncalls"] >= 100_000:
        return "very high call count — push the Python loop into a numpy vectorized op"
    return (
        "own-time hotspot in Python — vectorize / Numba-JIT the inner loop "
        "or cache repeated work"
    )


def _own_functions(modules):
    """Map (filename, first_lineno) -> function object for line-profiler lookup.

    Walks module-level functions and class methods so a hotspot identified by
    cProfile (keyed on filename + def-line) can be bridged to the live function
    object that ``line_profiler`` needs. Methods are unwrapped to their underlying
    function so ``@classmethod`` / ``@staticmethod`` and bound methods all register
    — a classmethod surfaces from :func:`inspect.getmembers` as a *bound method*
    (not a function), so a plain ``isfunction`` filter would silently drop it.
    """
    registry = {}

    def add(obj):
        # Unwrap descriptors/bound methods to the underlying function object —
        # the descriptor itself has no __code__, so without this every classmethod
        # (and property) would be silently dropped from the drill-down.
        if isinstance(obj, (staticmethod, classmethod)):
            obj = obj.__func__
        elif inspect.ismethod(obj):  # bound method, e.g. a classmethod on the class
            obj = obj.__func__
        elif isinstance(obj, property):
            obj = obj.fget
        elif isinstance(obj, functools.cached_property):
            obj = obj.func
        code = getattr(obj, "__code__", None)
        if code is not None:
            registry[(code.co_filename, code.co_firstlineno)] = obj

    for mod in modules:
        for _, obj in inspect.getmembers(mod):
            if inspect.isfunction(obj):
                add(obj)
            elif inspect.isclass(obj):
                # vars() exposes static/class methods as their raw descriptors;
                # getmembers resolves them (classmethods -> bound methods). Scan
                # both so every method kind is captured and unwrapped.
                for member in list(vars(obj).values()):
                    add(member)
                for _, member in inspect.getmembers(obj):
                    add(member)
    return registry


def _docstring_lines(funcs):
    """Absolute ``(filename, lineno)`` pairs occupied by each function's docstring.

    Used to drop docstring rows from the line-profiler drill-down: they are never
    executed, so they carry no timing and only pad the output.
    """
    pairs = set()
    for f in funcs:
        try:
            src_lines, start = inspect.getsourcelines(f)
            node = ast.parse(textwrap.dedent("".join(src_lines))).body[0]
        except (OSError, TypeError, SyntaxError, IndexError):
            continue
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if ast.get_docstring(node, clean=False) is None:
            continue
        doc = node.body[0]  # the docstring is the first body statement
        filename = f.__code__.co_filename
        for rel in range(doc.lineno, (doc.end_lineno or doc.lineno) + 1):
            pairs.add((filename, start + rel - 1))
    return pairs


def _strip_docstrings(text, funcs):
    """Remove docstring rows from line-profiler output (never executed, all noise)."""
    doc = _docstring_lines(funcs)
    if not doc:
        return text
    kept = []
    cur_file = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("File:"):
            cur_file = stripped[len("File:") :].strip()
        head = line.split(maxsplit=1)
        if cur_file and head and head[0].isdigit():
            if (cur_file, int(head[0])) in doc:
                continue
        kept.append(line)
    return "\n".join(kept)


def _line_drilldown(call, args, kwargs, funcs):
    if not funcs:
        return ""
    lp = LineProfiler()
    for f in funcs:
        lp.add_function(f)
    lp.runcall(call, *args, **(kwargs or {}))
    buf = io.StringIO()
    lp.print_stats(stream=buf, output_unit=1e-3)
    return _strip_docstrings(buf.getvalue(), funcs)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _fmt_func(r):
    return f"`{_module_label(r['file'])}:{r['name']}`"


def _render(title, total, rows, flags, line_text):
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = []
    w = out.append

    w(f"# Profiling report: {title}")
    w("")
    w(f"_Generated {ts} • total profiled own-time budget: {total:.3f} s_")
    w("")
    w(
        "> Auto-generated by `tests/profiling/_profiling.py`. Each KPF method is "
        "charged the own time of the library/builtin calls it drives "
        "(**attributed time**), so "
        "the ranking points at the method to optimize rather than the library leaf "
        f"it bottoms out in. A **hotspot** has ≥ {HOTSPOT_FRACTION:.0%} attributed "
        f"time and ≥ {HOTSPOT_MIN_SECONDS:.0f} s; the line-level drill-down shows "
        "where inside it the time goes."
    )
    w("")

    # Recommended actions --------------------------------------------------
    w(
        f"## Recommended actions (≥ {HOTSPOT_FRACTION:.0%} attributed time "
        f"and ≥ {HOTSPOT_MIN_SECONDS:.0f} s)"
    )
    w("")
    if flags:
        for r in flags:
            w(
                f"- {_fmt_func(r)} — {r['attr_frac']:.1%} "
                f"({r['attr_s']:.3f} s): {_note(r)}"
            )
    else:
        w("_Nothing over threshold — **no action needed**._")
    w("")

    # KPF-method ranking ---------------------------------------------------
    w(f"## KPF methods ≥ {TOP_FUNCTION_MIN_FRACTION:.0%} of runtime (attributed)")
    w("")
    w("| method | attributed % | attributed s | own s | calls |")
    w("|---|---:|---:|---:|---:|")
    for r in rows:
        if r["attr_frac"] < TOP_FUNCTION_MIN_FRACTION:
            continue
        w(
            f"| {_fmt_func(r)} | {r['attr_frac']:.1%} | {r['attr_s']:.3f} | "
            f"{r['own_s']:.3f} | {r['ncalls']} |"
        )
    w("")
    covered = sum(r["attr_s"] for r in rows)
    w(
        f"_KPF methods account for {covered:.3f} s ({covered / total:.0%}) of the "
        f"{total:.3f} s budget; the rest is library/driver code with no KPF caller "
        "(e.g. test setup, I/O)._"
    )
    w("")

    # Line drill-down ------------------------------------------------------
    if line_text:
        w("## Line-level drill-down (hotspot methods)")
        w("")
        w("```")
        w(line_text.rstrip())
        w("```")
        w("")

    return "\n".join(out)


def _render_recipe(title, summary, split=None, quicklook=None):
    """Render an end-to-end recipe report.

    Just the cumulative **High-level summary** (wall-clock per pipeline stage, a
    clean partition of the recipe runtime; :func:`_recipe_summary`) plus the
    separately-reported network/barycentric totals. Function-level detail
    (tentpoles, flags, line drill-down) is relegated to the per-module reports
    (:func:`_render`). Network and barycentric correction are quarantined because
    they are nondeterministic.

    ``split`` (optional, from :func:`_io_compute_split`) adds a whole-run I/O-vs-
    compute breakdown — used by the masters recipe, whose I/O lives *inside* the
    modules and so is not otherwise visible in the per-stage summary.
    ``quicklook`` (optional, from :func:`_quicklook_breakdown`) adds a per-plot
    breakdown of the quicklook stage; rendered only when non-empty (science).
    """
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    denom = summary["denominator"]
    has_bary = summary["bary_seconds"] > 0
    out = []
    w = out.append

    w(f"# Profiling report: {title}")
    w("")
    excl = " (excl. barycentric)" if has_bary else ""
    w(f"_Generated {ts} • recipe wall-clock{excl}: {denom:.3f} s._")
    w("")
    quarantined = (
        "The barycentric-correction stage and network time are reported "
        "separately (excluded from the rows) because they are nondeterministic. "
        if has_bary
        else "Network time is reported separately because it is nondeterministic. "
    )
    w(
        "> Auto-generated by `tests/profiling/_profiling.py`. The High-level "
        "summary is cumulative wall-clock per pipeline stage (each stage includes "
        "every function it calls), partitioned so the rows sum to the recipe runtime. "
        f"{quarantined}Function-level detail is in the per-module reports."
    )
    w("")

    # High-level summary (cumulative wall-clock per stage) -----------------
    w("## High-level summary")
    w("")
    w("| stage / bucket | % wall | s |")
    w("|---|---:|---:|")
    for lbl, secs, frac, _kind in summary["rows"]:
        w(f"| `{lbl}` | {frac:.1%} | {secs:.3f} |")
    w("")

    # Quarantined (nondeterministic) totals, reported separately ----------
    excluded = []
    if summary["bary_seconds"] > 0:
        excluded.append(
            f"- **Barycentric correction stage** — {summary['bary_seconds']:.3f} s "
            "wall-clock (excluded; triggers a nondeterministic IERS network fetch)."
        )
    if summary["network_seconds"] > 0:
        whence = "mostly inside barycentric" if has_bary else "not disk I/O"
        excluded.append(
            f"- **Network** (SSL/socket) — {summary['network_seconds']:.3f} s own "
            f"time (excluded; nondeterministic, {whence})."
        )
    w("**Excluded from the summary above (reported separately):**")
    w("")
    out.extend(excluded or ["- _None — fully deterministic disk + compute._"])
    w("")

    # I/O vs compute (whole run; masters recipe) --------------------------
    if split is not None:
        base = split["total"]
        io_frac = split["io"] / base
        w("## I/O vs compute (whole run)")
        w("")
        w(
            "I/O is the cumulative time of the data-model read/write calls "
            "(`from_fits` / `to_fits`) — where every disk read and write happens; "
            "compute is the rest of the budget (including the image assembly run "
            "while loading each frame)."
        )
        w("")
        w("| category | % | s |")
        w("|---|---:|---:|")
        w(f"| compute | {1 - io_frac:.1%} | {split['compute']:.3f} |")
        w(f"| I/O (disk read/write) | {io_frac:.1%} | {split['io']:.3f} |")
        w("")
        verdict = "compute-dominated" if io_frac < 0.5 else "I/O-dominated"
        w(
            f"_Runtime is **{verdict}** "
            f"({1 - io_frac:.0%} compute / {io_frac:.0%} I/O)._"
        )
        w("")

    # Quicklook plot breakdown (science recipe) ---------------------------
    if quicklook and quicklook["rows"]:
        w("## Quicklook plot breakdown")
        w("")
        w(
            f"Wall-clock per plot within the quicklook stage "
            f"({quicklook['total']:.3f} s total; % of that). Each plot includes "
            "its matplotlib render and PNG save."
        )
        w("")
        w("| plot | % | s |")
        w("|---|---:|---:|")
        for lbl, secs, frac in quicklook["rows"]:
            w(f"| `{lbl}` | {frac:.1%} | {secs:.3f} |")
        w("")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_profile(
    title,
    report_name,
    setup,
    call,
    candidate_modules,
    kwargs=None,
    line_pass=True,
    recipe=False,
    io_compute=False,
):
    """Profile ``call(*setup())`` and write a tentpole report.

    Parameters
    ----------
    title : str
        Human-readable report title.
    report_name : str
        Base filename for the Markdown report under :data:`REPORTS_DIR`.
    setup : callable
        Zero-arg builder returning the args for ``call`` (a single value or a
        tuple). Called once per profiling pass so each pass gets fresh input;
        keep expensive input construction here so it is excluded from the
        measured call.
    call : callable
        The function actually profiled, invoked as ``call(*args, **kwargs)``.
    candidate_modules : list
        Modules whose functions are eligible for line-level drill-down.
    kwargs : dict, optional
        Extra keyword args forwarded to ``call``.
    line_pass : bool, default True
        Run the second :mod:`line_profiler` pass over the tentpoles. Disable for
        whole-pipeline harnesses where re-running everything is too costly and
        the per-module reports already provide line detail.
    recipe : bool, default False
        Use the end-to-end recipe report layout: a cumulative High-level summary
        of wall-clock per pipeline stage (quicklook and QC/diagnostics collapsed
        across levels), with network and barycentric correction quarantined and
        reported separately. The two ``profile_*_recipe.py`` harnesses set this;
        per-module harnesses use the default own-time layout. Implies no line
        pass, and replaces the function-level sections (those live per module).
    io_compute : bool, default False
        Add a whole-run I/O-vs-compute breakdown below the summary (recipe only).
        For the masters recipe, whose disk I/O lives *inside* the product modules
        and so is not a visible row in the per-stage summary.
    """
    require_testdata()
    print(f"\n=== profiling: {title} ===\n")

    def _as_args(value):
        return value if isinstance(value, tuple) else (value,)

    # Pass 1: cProfile on the isolated call.
    args = _as_args(setup())
    pr = cProfile.Profile()
    pr.enable()
    call(*args, **(kwargs or {}))
    pr.disable()

    stats = pstats.Stats(pr)

    if recipe:
        rows, _total = _rank(stats)
        summary = _recipe_summary(stats, call, _network_own(rows))
        split = _io_compute_split(stats) if io_compute else None
        quicklook = _quicklook_breakdown(stats)
        report = _render_recipe(title, summary, split, quicklook)
        print(report)
        return _write_report(report, report_name)

    # Per-module report: attribute time to the KPF method that drives it.
    rows, total = _kpf_rows(stats)
    flags = _flags(rows)

    # Pass 2: line-level drill-down of each hotspot method.
    line_text = ""
    if line_pass and flags:
        registry = _own_functions(candidate_modules)
        hot_funcs = [
            registry[(r["file"], r["line"])]
            for r in flags
            if (r["file"], r["line"]) in registry
        ]
        if hot_funcs:
            line_text = _line_drilldown(call, _as_args(setup()), kwargs, hot_funcs)

    report = _render(title, total, rows, flags, line_text)
    print(report)
    return _write_report(report, report_name)


def _write_report(report, report_name):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"{report_name}.md"
    out_path.write_text(report + "\n")
    print(f"\n[report written] {out_path}")
    return out_path
