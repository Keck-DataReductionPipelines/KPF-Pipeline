# Order-trace references

Vetted order traces, one per stable stretch of instrument geometry. Spectral
extraction reads them: `SpectralExtraction._read_order_trace_reference` picks
the most recent file measured **before** the frame and **within the frame's
instrument era** (`reference/instrument_eras.csv`), so a file takes effect on
its own date and stays in force until the next one in that era.

Files are named `order_trace_<datecode>.csv`, the datecode of the master flat
they were measured from. Each is a `make_master` output of
`kpfpipe.modules.masters.order_trace`, copied unchanged: one row per expected
trace with `Chip`, `Fiber`, `Order`, `Coeff0..3`, `BottomEdge`, `TopEdge`,
`X1`, `X2`, `PolyfitRMS`, `Status`.

## Contents

| Era | File | Source epoch | Role |
|-----|------|--------------|------|
| 1.0 | `order_trace_20221207.csv` | KP.20221207.71923.90 | era start |
| 1.0 | `order_trace_20230406.csv` | KP.20230406.70700.10 | after jump 1 |
| 1.0 | `order_trace_20230521.csv` | KP.20230521.73958.37 | after jump 2 |
| 1.0 | `order_trace_20230623.csv` | KP.20230623.80896.19 | after jump 3 |
| 1.0 | `order_trace_20230731.csv` | KP.20230731.81094.75 | after jump 4 |
| 2.0 | `order_trace_20240224.csv` | KP.20240224.00049.95 | era start |
| 2.0 | `order_trace_20240528.csv` | KP.20240528.81096.73 | after jump |
| 2.6 | `order_trace_20250102.csv` | KP.20250102.00031.42 | era start |
| 2.6 | `order_trace_20250117.csv` | KP.20250117.81096.88 | after settling |
| 2.6 | `order_trace_20250225.csv` | KP.20250225.81096.24 | after jump |
| 3.0 | `order_trace_20250424.csv` | KP.20250424.81096.58 | era start |
| 3.0 | `order_trace_20250531.csv` | KP.20250531.00066.95 | after settling |
| 4.0 | `order_trace_20251030.csv` | KP.20251030.81091.48 | era start |
| 4.0 | `order_trace_20251122.csv` | KP.20251122.82590.83 | after jump |
| 4.0 | `order_trace_20260327.csv` | KP.20260327.82610.51 | after jump |

Roles:

- **era start** — the era's first available trace. Required: without it the
  era's opening nights resolve to no reference at all.
- **after settling** — eras 2.6 and 3.0 open with the traces decaying to their
  settled position over 1-2 weeks. These files are the first epoch already at
  the settled level.
- **after jump** — the traces stepped to a new position between two
  consecutive flats and stayed there. Each file is the first epoch on the new
  side of the step.

## Provenance

Selected from 1899 order traces over 960 nights (every master flat that traced
successfully, 2022-12 to 2026-04) by a one-off stability analysis that measures
each epoch's centerlines against the era's median centerline. Within an era and
away from the jumps below, the traces hold to 0.01-0.08 pixel RMS, so one
reference per stretch is enough.

That analysis was exploratory and is not kept in the repo; the numbers it
produced are recorded here, and re-running it means re-deriving it from the
master flats. Jumps came from the same pass: a level change of at least 0.025
pixel that is at least 15x the local scatter, measured across the 8 epochs
either side, per chip. Detected jumps, with the chip that moved:

| Era | Jump | Chip | Step [pix] |
|-----|------|------|-----------|
| 1.0 | 2023-04-06 | RED (GREEN also moves) | 0.117 |
| 1.0 | 2023-05-21 | RED | 0.127 |
| 1.0 | 2023-06-23 | GREEN | 0.065 |
| 1.0 | 2023-07-31 | RED | 0.026 |
| 2.0 | 2024-05-28 | GREEN and RED | 0.038 / 0.033 |
| 2.6 | 2025-02-25 | GREEN | 0.074 |
| 4.0 | 2025-11-22 | GREEN and RED | 0.121 / 0.067 |
| 4.0 | 2026-03-27 | GREEN and RED | 0.113 / 0.063 |

Era 3.0 has no jump.

## Known limits

- **Era 1.0 before 2022-12-07 has no reference.** The era opens 2022-11-09, but
  the first nights either lack a master flat or produced an unusable one, so
  frames from 2022-11-09 to 2022-12-06 raise `FileNotFoundError` in spectral
  extraction.
- **A file dated `<datecode>` is timestamped at 00:00 UT of that date**, while
  several of these traces come from that night's *evening* flat. Where a jump
  happened between a night's morning and evening flat (2023-07-31, 2025-02-25,
  2026-03-27), frames taken earlier in that same night resolve to the new file
  even though they were taken before the step.
- **Era 1.0 traces carry GREEN SKY order 0 as `missing`** — that orderlet is
  off the detector in era 1.0, so those files hold 174 GREEN traces, not 175.
  Every file also carries 2-3 `partial` edge traces (GREEN SKY 0, GREEN CAL 34,
  RED SKY 0), fitted over less than the full column range.
- **Era 1.0 RED flats alternate between two trace states** night to night
  (roughly 0.03-0.04 vs 0.11-0.20 pixel RMS against the era median),
  independent of the jumps above. For 2023-04-06 the least deviant of that
  night's three flats was taken; the choice moves RED by about 0.08 pixel.
