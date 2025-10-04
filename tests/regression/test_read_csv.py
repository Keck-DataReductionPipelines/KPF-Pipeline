# tests/regression/test_read_csv.py
import csv
from pathlib import Path

import pandas as pd
import pytest

# Directories with .csv files to validate
DIRS = [
    '/code/KPF-Pipeline/static/',
    '/code/KPF-Pipeline/caldates/',
]

CALDATES_DIR = Path('/code/KPF-Pipeline/caldates/')
TSDB_DIR     = Path('/code/KPF-Pipeline/static/tsdb_tables/')

EXPECTED_KPFERA_COLS = ["KPFERA", "UT_start_date", "UT_end_date", "comment"]
EXPECTED_CALDATES_COLS_REQUIRED = ["CALTAG", "UT_start_date", "UT_end_date", "CALPATH"]
EXPECTED_CALDATES_COLS_WITH_COMMENT = EXPECTED_CALDATES_COLS_REQUIRED + ["comment"]

EXPECTED_TSDB_TABLES_COLS  = ["keyword", "datatype", "description", "unit"]
EXPECTED_TSDB_HEADER_LINE  = "keyword|datatype|description|unit"


def list_csv_files(dirs, recursive=True):
    paths = []
    for d in dirs:
        p = Path(d).expanduser()
        if not p.is_dir():
            continue
        it = p.rglob("*.csv") if recursive else p.glob("*.csv")
        paths.extend(it)
    # Deduplicate + sort
    return sorted(dict.fromkeys(map(lambda x: x.resolve(), paths)))


def read_csv_strict_python(path, sep=",", *, keep_default_na=True):
    """Strict CSV read: raises on short/long rows or bad quoting."""
    return pd.read_csv(
        path,
        sep=sep,
        engine="python",
        on_bad_lines="error",
        skipinitialspace=True,
        dtype="string",            # read everything as text first; enforce types later
        keep_default_na=keep_default_na,  # allow overriding NA inference (e.g., keep 'None' literal)
    )


def _first_line_equals(path: Path, target: str) -> bool:
    """
    Return True if the file's very first line equals `target` (whitespace-trimmed).
    Uses utf-8-sig to gracefully handle BOMs.
    """
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            first = f.readline()
    except Exception:
        return False
    return first.strip() == target


def _assert_fixed_columns(path: Path, expected_cols: int, delimiter: str = "|"):
    """Raise if any data row has a different number of columns than expected (includes line numbers)."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter, skipinitialspace=True)
        # Skip header
        _ = next(reader, None)
        for i, row in enumerate(reader, start=2):  # 1-based; header is line 1
            if len(row) != expected_cols:
                raise AssertionError(
                    f"{path}: line {i} has {len(row)} fields; expected {expected_cols}"
                )


def _lines_from_mask(mask: pd.Series) -> list[int]:
    """Return 1-based line numbers (header=1) for True entries in mask."""
    # Data rows start at line 2
    return (mask[mask].index.to_series().astype(int) + 2).tolist()


# ---------- Generic well-formedness ----------
@pytest.mark.parametrize("csv_path", list_csv_files(DIRS))
def test_csv_is_well_formed(csv_path: Path):
    # Use pipe for files under tsdb_tables/, comma otherwise
    sep = "|" if "tsdb_tables" in str(csv_path) else ","
    # Using Python engine ensures malformed lines (e.g., missing comma/pipe) raise
    read_csv_strict_python(csv_path, sep)


# ---------- kpfera_definitions.csv schema ----------
def _validate_kpfera_schema(csv_path: Path):
    # Strict structural read first (catches missing/trailing commas)
    df = read_csv_strict_python(csv_path, sep=",")

    # Exact columns (order and names)
    cols = [c.strip() for c in df.columns]
    if cols != EXPECTED_KPFERA_COLS:
        raise AssertionError(f"{csv_path}: columns must be {EXPECTED_KPFERA_COLS}, found {cols}")

    # Types with line-numbered errors
    # KPFERA numeric
    kpf = pd.to_numeric(df["KPFERA"], errors="coerce")
    bad_kpf = kpf.isna()
    if bad_kpf.any():
        lines = _lines_from_mask(bad_kpf)
        raise AssertionError(f"{csv_path}: KPFERA not numeric at lines {lines}")

    # Datetimes exact format
    for col in ["UT_start_date", "UT_end_date"]:
        parsed = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        bad = parsed.isna()
        if bad.any():
            lines = _lines_from_mask(bad)
            raise AssertionError(f"{csv_path}: {col} not in 'YYYY-MM-DD HH:MM:SS' at lines {lines}")
        df[col] = parsed

    # comment: coerce to string (no strict assertion)
    df["comment"] = df["comment"].astype("string")

    # Logical ordering with line numbers
    bad_order = df["UT_start_date"] > df["UT_end_date"]
    if bad_order.any():
        lines = _lines_from_mask(bad_order)
        raise AssertionError(f"{csv_path}: UT_start_date > UT_end_date at lines {lines}")


@pytest.mark.parametrize(
    "csv_path",
    [p for p in list_csv_files(DIRS) if Path(p).name == "kpfera_definitions.csv"]
)
def test_kpfera_definitions_schema(csv_path: Path):
    try:
        _validate_kpfera_schema(csv_path)
    except Exception as e:
        pytest.fail(f"{e}")


# ---------- /caldates/ schema (comment optional) ----------
def _validate_caldates_schema(csv_path: Path):
    # Strict structural read first
    df = read_csv_strict_python(csv_path, sep=",")

    REQUIRED = ["CALTAG", "UT_start_date", "UT_end_date", "CALPATH"]
    cols = [c.strip() for c in df.columns]
    if cols not in (REQUIRED, REQUIRED + ["comment"]):
        raise AssertionError(f"{csv_path}: columns must be {REQUIRED} (optionally + 'comment'), found {cols}")

    # Types with line-numbered errors
    caltag = pd.to_numeric(df["CALTAG"], errors="coerce")
    bad_tag = caltag.isna()
    if bad_tag.any():
        lines = _lines_from_mask(bad_tag)
        raise AssertionError(f"{csv_path}: CALTAG not numeric at lines {lines}")

    for col in ["UT_start_date", "UT_end_date"]:
        parsed = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        bad = parsed.isna()
        if bad.any():
            lines = _lines_from_mask(bad)
            raise AssertionError(f"{csv_path}: {col} not in 'YYYY-MM-DD HH:MM:SS' at lines {lines}")
        df[col] = parsed

    bad_order = df["UT_start_date"] > df["UT_end_date"]
    if bad_order.any():
        lines = _lines_from_mask(bad_order)
        raise AssertionError(f"{csv_path}: UT_start_date > UT_end_date at lines {lines}")

    # CALPATH must be a non-empty string
    bad_calpath = df["CALPATH"].isna() | (df["CALPATH"].str.strip() == "")
    if bad_calpath.any():
        lines = _lines_from_mask(bad_calpath)
        raise AssertionError(f"{csv_path}: CALPATH missing/empty at lines {lines}")

    # comment is OPTIONAL — if present, coerce to string (don’t assert dtype)
    if "comment" in df.columns:
        df["comment"] = df["comment"].astype("string")


@pytest.mark.parametrize(
    "csv_path",
    sorted(p for p in CALDATES_DIR.rglob("*.csv") if p.is_file())
)
def test_caldates_schema(csv_path: Path):
    try:
        _validate_caldates_schema(csv_path)
    except Exception as e:
        pytest.fail(f"{e}")


# ---------- /static/tsdb_tables/ schema (apply only when header matches) ----------
def _validate_tsdb_tables_schema(csv_path: Path):
    # 1) Raw field-count check catches missing/extra pipes on any line (with line numbers)
    _assert_fixed_columns(csv_path, expected_cols=4, delimiter="|")

    # 2) Parse strictly with pandas BUT do NOT treat 'None' as NA
    df = read_csv_strict_python(csv_path, sep="|", keep_default_na=False)

    # Exact header match (names & count)
    cols = [c.strip() for c in df.columns]
    if cols != EXPECTED_TSDB_TABLES_COLS:
        raise AssertionError(
            f"{csv_path}: columns must be {EXPECTED_TSDB_TABLES_COLS} (found {cols})"
        )

    # No duplicate header names
    if len(set(cols)) != len(cols):
        raise AssertionError(f"{csv_path}: duplicate column names detected: {cols}")

    # Required fields must be non-empty strings (report exact lines)
    for required in ["keyword", "datatype", "description", "unit"]:
        bad = df[required].isna() | (df[required].str.strip() == "")
        if bad.any():
            lines = _lines_from_mask(bad)
            raise AssertionError(f"{csv_path}: '{required}' missing/empty at lines {lines}")


@pytest.mark.parametrize(
    "csv_path",
    sorted(p for p in TSDB_DIR.rglob("*.csv") if p.is_file())
)
def test_tsdb_tables_schema(csv_path: Path):
    # Only validate files whose *first line* exactly matches the tsdb header
    if not _first_line_equals(csv_path, EXPECTED_TSDB_HEADER_LINE):
        pytest.skip(f"Skipping {csv_path.name}: header does not match '{EXPECTED_TSDB_HEADER_LINE}'")
    try:
        _validate_tsdb_tables_schema(csv_path)
    except Exception as e:
        pytest.fail(str(e))
