"""End-of-run summary formatters shared by the KPF recipes.

Pure string builders for the compact run-verdict block each recipe logs at
completion. They live here (not in ``kpfpipe/utils/logger.py``, which is
logging-setup infrastructure) so both recipes share one rendering while the
setup module stays free of domain content.
"""

import os


def science_run_summary(l4, elapsed_s):
    """Format the science recipe's end-of-run verdict from the finished L4.

    A compact, greppable roll-up read off the L4 the recipe just built: obs_id
    and the master-file and input/product paths from the RECEIPT provenance
    table, and the combined RV from PRIMARY, plus the
    wall-clock elapsed. ``RV``/``RVERR``/``BJDTDB`` that
    are not real numbers (absent or FITS UNDEFINED, e.g. no science combine ran)
    render as ``n/a``; ``RV`` is km/s (error in m/s), ``BJDTDB`` is BJD_TDB. The
    surrounding blank lines make the block stand out by eye in the log.
    """
    primary = l4.headers.get("PRIMARY", {})

    def base(path):
        return os.path.basename(path) if path else "n/a"

    def receipt_paths(function, arg_key):
        # Every matching row, unlike receipt_read_entry's most-recent one.
        table = getattr(l4, "receipt", None)
        if table is None or getattr(table, "empty", True):
            return []
        out = []
        for _, row in table.iterrows():
            if row.get("FUNCTION") != function:
                continue
            for token in str(row.get("ARGS", "")).split(", "):
                key, _, value = token.partition("=")
                if key.strip() == arg_key and value.strip():
                    out.append(value.strip())
        return out

    obs_id = getattr(l4, "obs_id", None) or primary.get("ORIGID") or "unknown"
    cal = l4.receipt_read_entry("calibration_association")
    masters = {
        "bias": cal.get("biasfile"),
        "dark": cal.get("darkfile"),
        "thar": cal.get("wlsfile"),
    }
    masters_str = "  ".join(f"{k}={base(v)}" for k, v in masters.items())
    # First from_fits is the original L0 read; a reload would append its own.
    inputs = receipt_paths("from_fits", "fn")[:1]
    outputs = receipt_paths("to_fits", "out_filepath")
    inputs_str = "  ".join(base(p) for p in inputs) or "n/a"
    outputs_str = "  ".join(base(p) for p in outputs) or "n/a"

    rv, rverr, bjd = primary.get("RV"), primary.get("RVERR"), primary.get("BJDTDB")
    if isinstance(rv, (int, float)):
        err = f"{rverr * 1e3:.3f}" if isinstance(rverr, (int, float)) else "n/a"
        bjd_str = f"{bjd:.6f}" if isinstance(bjd, (int, float)) else "n/a"
        rv_line = f"{rv:+.5f} km/s  err {err} m/s  @ BJD_TDB {bjd_str}"
    else:
        rv_line = "n/a (no science combine)"

    lines = [
        f"===== run summary: {obs_id} =====",
        f"  inputs:   {inputs_str}",
        f"  masters:  {masters_str}",
        f"  outputs:  {outputs_str}",
        f"  RV:       {rv_line}",
        f"  elapsed:  {elapsed_s:.1f} s",
    ]
    return "\n\n" + "\n".join(lines) + "\n\n"


def masters_run_summary(datecode, built, elapsed_s):
    """Format the masters recipe's end-of-run verdict.

    ``built`` is the list of ``(cal_type, path, n_frames)`` stacked this run
    (empty if none) -- masters have no single product to read back, so the
    recipe passes what it stacked. The surrounding blank lines make the block
    stand out by eye in the log.
    """
    lines = [f"===== masters run summary: {datecode} ====="]
    if built:
        for cal_type, path, n_frames in built:
            name = os.path.basename(path) if path else "n/a"
            lines.append(f"  {cal_type:<6s} {name}  ({n_frames} frames)")
    else:
        lines.append("  (no masters built)")
    lines.append(f"  elapsed:  {elapsed_s:.1f} s")
    return "\n\n" + "\n".join(lines) + "\n\n"
