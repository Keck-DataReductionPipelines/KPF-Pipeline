# Single-Wideflat Order Trace Module

## Summary

Create a clean vNext `OrderTrace` module that reads one raw L0 wideflat, performs overscan removal and bias subtraction, discovers and measures both CCDs’ traces, and always writes pipeline-compatible GREEN and RED CSV files. Legacy code supplies scientific ideas only; the implementation follows current repository architecture and style.

## Public Interface

```python
tracer = OrderTrace(wideflat_filename, config=None)

tables = tracer.perform(
    chips=None,
    output_dir=output_dir,
    cal_order3_y=None,
    overwrite=False,
)
```

- `chips=None` processes GREEN and RED; an explicit subset is supported.
- `output_dir` is required. Requested chips are written as `order_trace_green.csv` and `order_trace_red.csv`.
- `cal_order3_y` is an optional mapping such as `{"GREEN": 272.5, "RED": 281.0}`. Values are CAL-fiber/order-3 row positions at assembled detector column `x=0`.
- Existing outputs raise `FileExistsError` unless `overwrite=True`.
- The returned mapping contains the same DataFrames written to disk.
- CSV columns and index match the current references exactly: `Coeff0–Coeff3`, `BottomEdge`, `TopEdge`, `X1`, `X2`, `Fiber`, `Order`. Coefficients use `y = Coeff0 + Coeff1*x + Coeff2*x² + Coeff3*x³`.

## Implementation

- Add `kpfpipe/modules/order_trace.py` as a standalone class following the required constructor, method ordering, configuration, logging, receipt, and `info()` conventions.
- Load the raw L0 filename with `KPF0`, run `ImageAssembly`, associate a bias through `CalibrationAssociation`, and run `ImageProcessing` with bias enabled and dark/flat disabled.
- Resolve the observation’s `INSTERA` from `DATE-OBS`. Update `reference/kpf_instrument_eras.csv` from the newer legacy definitions while retaining the vNext `INSTERA` schema.
- For now, every known era uses `reference/order_trace_green.csv` and `reference/order_trace_red.csv` as approximate seed geometry. No new manifest is added.
- If the observation is outside a defined era, require `cal_order3_y` for every requested chip. Also allow these anchors to override initialization for a known era.
- Translate each chip’s seed template vertically so its CAL/order-3 curve matches the supplied row at `x=0`. Templates provide search windows and fiber/order identity only; their fitted values are never copied into the result.
- Port the useful legacy discovery concepts: background removal, smoothing, illuminated-pixel/cluster detection, rejection of noise and broken clusters, and association of detected traces with the predicted order grid.
- Measure centers using the notebook method: median column strips, percentile background subtraction, Gaussian smoothing, winsorized flux centroids, and sequential trace following. Sample 65 detector columns by default.
- Fit each centerline with an iterative four-sigma MAD-rejected cubic, requiring at least half the sampled positions to survive and rejecting crossing or incorrectly ordered traces.
- Measure `BottomEdge` and `TopEdge` from robust cross-dispersion profiles, using the legacy Gaussian-width and neighboring-valley constraints with a two-pixel orderlet gap. Derive `X1` and `X2` from valid measured coverage.
- Label traces in detector-row order as SKY, SCI1, SCI2, SCI3, CAL for each order. Omit a trace only when its predicted and measured coverage is wholly off-detector, preserving the existing 175-row GREEN and 159-row RED behavior for the 2024 reference era.
- Validate both requested tables completely before writing either file, then write them atomically.
- Add `[MODULE_ORDER_TRACE]` configuration to the masters configuration with the notebook/legacy defaults for sampling, search windows, rejection, polynomial degree, width estimation, and reference paths.
- Do not restore the stashed implementation, modify the notebook, change `SpectralExtraction`, or integrate the module into recipes/CLI in this phase.

## Test Plan

- Synthetic curved-trace tests verify recovery of centers to within one pixel, positive measured widths, correct horizontal bounds, canonical labeling, non-crossing traces, and cubic coefficient orientation.
- Test known-era lookup, era-table gaps, unknown-era failure without anchors, per-chip manual anchor translation, and chip-subset validation.
- Mock the preprocessing modules to verify raw L0 loading, image assembly, bias association/subtraction, and explicit exclusion of dark/flat corrections.
- Verify exact CSV schema, row ordering, filenames, round-trip values, overwrite protection, and compatibility with `SpectralExtraction`.
- Add a slow, testdata-gated run for `KP.20240923.00139.44`, expecting 175 GREEN and 159 RED rows and sub-pixel fit residuals against measured centers.

## Assumptions

- GREEN uses `reference/order_trace_green.csv`; RED uses `reference/order_trace_red.csv` for every currently defined era.
- Manual positions refer to the post-assembly, overscan-removed detector orientation at column zero.
- New-era execution generates candidate CSVs but does not automatically edit reference files or register a new era.
- Recipe and CLI integration will be planned after the standalone module is scientifically validated.
