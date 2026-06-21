"""Shared profiling harness for the KPF-DRP vNext "tallest tentpole" suite.

This is **not** a test module (no ``test_`` prefix) and pytest does not collect
it. It is imported by the standalone ``tests/profile_*.py`` harnesses, which are
run on demand via ``make profile*`` (see the ``## Profiling`` section of
CLAUDE.md). It runs fully standalone with no interactive input.

Strategy
--------
We only care about the most critical bottlenecks ("tentpoles"). Each harness:

1. Runs the target under :mod:`cProfile` (pass 1) to rank *every* function call
   by own time (``tottime``) and cumulative time (``cumtime``).
2. Detects the **tentpoles** — every own-code function contributing more than
   :data:`TENTPOLE_FRACTION` of the own-time budget (so 1-3 dominant hotspots,
   not just the single biggest, which is always included).
3. Drills into each tentpole line-by-line with :mod:`line_profiler` (pass 2).
4. Emits a human-readable report to stdout *and* to a gitignored Markdown file
   under :data:`REPORTS_DIR`, auto-flagging anything over the thresholds with a
   generic, rule-based optimization note.

Profiling is meaningful only at realistic array sizes, so the harnesses run on
the real (gitignored) frames in ``tests/testdata`` and skip cleanly when those
frames are absent (mirrors the ``requires_testdata`` test pattern).
"""

import cProfile
import inspect
import io
import pstats
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from line_profiler import LineProfiler

# --- paths and sample frames -----------------------------------------------

TESTDATA_DIR = Path(__file__).parent / "testdata"
REPORTS_DIR = Path(__file__).parent / "profiling" / "reports"
CONFIGS_DIR = Path(__file__).parent.parent / "configs"

# Canonical sample frames present in tests/testdata/{L0,masters}/20240405.
SCIENCE_OBS_ID = "KP.20240405.40113.57"
MASTERS_DATECODE = "20240405"

# --- thresholds (fractions are of the total own-time budget) ----------------

TENTPOLE_FRACTION = 0.25  # > this share of own time -> a "tentpole"
FLAG_FRACTION = 0.05  # >= this share of own time -> flagged
FLAG_ABS_SECONDS = 1.0  # >= this many seconds of own time -> flagged
MODULE_FLAG_FRACTION = 0.10  # >= this share for a whole module -> flagged
TOP_N = 25  # rows in the ranking table


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
    from kpfpipe.utils.io import build_filepath

    return KPF0.from_fits(
        build_filepath(SCIENCE_OBS_ID, "L0", data_root=str(TESTDATA_DIR))
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

# The bundled darks span two default-gap clusters, so we widen the gap to group
# the five frames into one stack — the same accommodation the master-dark
# regression test makes (see tests/test_master_dark.py). Harmless for bias/thar,
# which already form a single cluster.
MASTERS_CLUSTER_GAP_SECONDS = 24 * 3600

# Reading masters (bias/dark/flat/thar) for the dark and WLS harnesses, which
# associate pre-built masters rather than constructing them. The bundled masters
# live under tests/testdata/masters, addressed via KPF_MASTERS_OUTPUT.
MASTERS_CONFIG = {"KPF_MASTERS_OUTPUT": str(TESTDATA_DIR)}


def masters_l0_files(imtype):
    """First real L0 cluster of ``imtype`` ('bias'|'dark'|'thar') from testdata."""
    from kpfpipe.utils.io import build_l0_file_lists

    l0_dir = str(TESTDATA_DIR / "L0" / MASTERS_DATECODE)
    return build_l0_file_lists(
        imtype, data_dir=l0_dir, cluster_gap_seconds=MASTERS_CLUSTER_GAP_SECONDS
    )[0]


# ---------------------------------------------------------------------------
# Profiling core
# ---------------------------------------------------------------------------


def _is_own_code(filename):
    """True for KPF pipeline source (not stdlib / site-packages)."""
    return "/kpfpipe/" in filename and "/site-packages/" not in filename


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
                "cum_frac": ct / total,
                "own": _is_own_code(fn),
            }
        )
    rows.sort(key=lambda r: r["tottime"], reverse=True)
    return rows, total


def _tentpoles(rows):
    """Own-code functions over the tentpole threshold; always at least one."""
    own = [r for r in rows if r["own"]]
    tps = [r for r in own if r["tot_frac"] > TENTPOLE_FRACTION]
    if not tps and own:
        tps = [own[0]]  # always surface the single tallest own-code hotspot
    return tps


def _flags(rows):
    return [
        r
        for r in rows
        if r["tot_frac"] >= FLAG_FRACTION or r["tottime"] >= FLAG_ABS_SECONDS
    ]


def _by_module(rows, total):
    """Aggregate own time per source module (plus a lumped library bucket)."""
    agg = {}
    for r in rows:
        label = _module_label(r["file"]) if r["own"] else "(library / numpy / I/O)"
        agg[label] = agg.get(label, 0.0) + r["tottime"]
    items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    return [(lbl, t, t / total) for lbl, t in items]


def _note(r):
    """Generic, rule-based optimization hint for a flagged row."""
    if not r["own"]:
        return "library/C call — reduce call count or use a better-batched API"
    if r["ncalls"] >= 100_000:
        return "very high call count — push the Python loop into a numpy vectorized op"
    if r["tot_frac"] > TENTPOLE_FRACTION:
        return (
            "dominant own-time hotspot — vectorize / Numba-JIT the inner loop "
            "or cache repeated work"
        )
    return "notable own-time — check for redundant computation or vectorization"


def _own_functions(modules):
    """Map (filename, first_lineno) -> function object for line-profiler lookup.

    Walks module-level functions and class methods so a tentpole identified by
    cProfile (keyed on filename + def-line) can be bridged to the live function
    object that ``line_profiler`` needs.
    """
    registry = {}

    def add(func):
        try:
            code = func.__code__
        except AttributeError:
            return
        registry[(code.co_filename, code.co_firstlineno)] = func

    for mod in modules:
        for _, obj in inspect.getmembers(mod):
            if inspect.isfunction(obj):
                add(obj)
            elif inspect.isclass(obj):
                for _, meth in inspect.getmembers(obj, inspect.isfunction):
                    add(meth)
    return registry


def _line_drilldown(call, args, kwargs, funcs):
    if not funcs:
        return ""
    lp = LineProfiler()
    for f in funcs:
        lp.add_function(f)
    lp.runcall(call, *args, **(kwargs or {}))
    buf = io.StringIO()
    lp.print_stats(stream=buf, output_unit=1e-3)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _fmt_func(r):
    where = _module_label(r["file"]) if r["own"] else r["file"].split("/")[-1]
    return f"`{where}:{r['name']}`"


def _render(title, total, rows, tentpoles, flags, modules, line_text):
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = []
    w = out.append

    w(f"# Profiling report: {title}")
    w("")
    w(f"_Generated {ts} • total profiled own-time budget: {total:.3f} s_")
    w("")
    w(
        "> Auto-generated by `tests/_profiling.py`. Tentpoles are own-code "
        f"functions over {TENTPOLE_FRACTION:.0%} of the budget. Curated, ranked "
        "recommendations live in the committed `PROFILING.md`."
    )
    w("")

    # Tentpoles ------------------------------------------------------------
    w(f"## Tentpoles (> {TENTPOLE_FRACTION:.0%} own time)")
    w("")
    if tentpoles:
        w("| function | own % | own s | cum % | calls |")
        w("|---|---:|---:|---:|---:|")
        for r in tentpoles:
            w(
                f"| {_fmt_func(r)} | {r['tot_frac']:.1%} | {r['tottime']:.3f} | "
                f"{r['cum_frac']:.1%} | {r['ncalls']} |"
            )
    else:
        w("_No own-code tentpole — time is dominated by library/I/O calls._")
    w("")

    # Module breakdown -----------------------------------------------------
    w("## Own time by module")
    w("")
    w("| module | own % | own s | flag |")
    w("|---|---:|---:|:--:|")
    for lbl, t, frac in modules[:TOP_N]:
        flag = "⚠️" if (frac >= MODULE_FLAG_FRACTION and "library" not in lbl) else ""
        w(f"| `{lbl}` | {frac:.1%} | {t:.3f} | {flag} |")
    w("")

    # Top functions --------------------------------------------------------
    w(f"## Top {TOP_N} functions by own time")
    w("")
    w("| function | own % | own s | cum s | calls |")
    w("|---|---:|---:|---:|---:|")
    for r in rows[:TOP_N]:
        w(
            f"| {_fmt_func(r)} | {r['tot_frac']:.1%} | {r['tottime']:.3f} | "
            f"{r['cumtime']:.3f} | {r['ncalls']} |"
        )
    w("")

    # Flags ----------------------------------------------------------------
    w(f"## Flags (≥ {FLAG_FRACTION:.0%} own time or ≥ {FLAG_ABS_SECONDS:.0f} s)")
    w("")
    if flags:
        for r in flags:
            w(
                f"- {_fmt_func(r)} — {r['tot_frac']:.1%} "
                f"({r['tottime']:.3f} s): {_note(r)}"
            )
    else:
        w("_Nothing over threshold — **no action needed**._")
    w("")

    # Line drill-down ------------------------------------------------------
    if line_text:
        w("## Line-level drill-down (tentpoles)")
        w("")
        w("```")
        w(line_text.rstrip())
        w("```")
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
    rows, total = _rank(stats)
    tentpoles = _tentpoles(rows)
    flags = _flags(rows)
    modules = _by_module(rows, total)

    # Pass 2: line-level drill-down of each tentpole.
    line_text = ""
    if line_pass:
        registry = _own_functions(candidate_modules)
        tp_funcs = [
            registry[(r["file"], r["line"])]
            for r in tentpoles
            if (r["file"], r["line"]) in registry
        ]
        if tp_funcs:
            line_text = _line_drilldown(call, _as_args(setup()), kwargs, tp_funcs)

    report = _render(title, total, rows, tentpoles, flags, modules, line_text)
    print(report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"{report_name}.md"
    out_path.write_text(report + "\n")
    print(f"\n[report written] {out_path}")
    return out_path
