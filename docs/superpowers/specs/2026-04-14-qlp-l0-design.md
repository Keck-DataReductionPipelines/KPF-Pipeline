# QLP L0 Design Spec

## Context

KPF-Pipeline v3 (kpf-next) can process L0 through L2 but has no quicklook plotting (QLP), diagnostics, or quality control (QC) infrastructure. The v2.12 pipeline has a comprehensive QLP system; we are porting it to v3 starting with L0-level plots.

This spec covers the first QLP module: L0 plots for development verification.

## Architecture decisions

**No Diagnostics module.** v2.12 had separate Diagnostics, QLP, and QC layers. In v3, Diagnostics is eliminated. Pipeline modules (ImageAssembly, etc.) compute and store all metrics in data products. QLP reads data products and makes plots. QC reads data products and makes pass/fail checks. No separate metric-computation layer.

**QLP does not compute.** QLP is pure visualization. It reads from data product extensions and headers. If a metric isn't in the data product, QLP doesn't show it. This ensures a clean separation: science logic lives in pipeline modules, display logic lives in QLP.

**Standalone first, recipe integration later.** QLP runs as a standalone script during development. Recipe integration is a future step.

## Scope

PlotL0 only. PlotL1, PlotL2, PlotL4, and QC are future work.

## Module structure

```
kpfpipe/qlp/
    __init__.py
    plot_l0.py          # PlotL0 class

scripts/
    qlp.py              # standalone CLI entry point

tests/
    test_qlp_l0.py      # structural tests
```

Future files (`plot_l1.py`, `plot_l2.py`, `plot_l4.py`) will follow the same pattern.

## PlotL0 class

File: `kpfpipe/qlp/plot_l0.py`

### Constructor

```python
class PlotL0:
    def __init__(self, l0_obj, output_dir=None):
        self.l0 = l0_obj
        self.output_dir = output_dir  # None = return Figure only, don't save
```

Takes a KPF0 object (raw CCD data). If `output_dir` is set, plots are saved as PNG files in addition to being returned.

### Method: `stitched_image(chip)`

Replicates `plot_L0_stitched_image` from v2.12 `modules/quicklook/src/analyze_l0.py`.

**Input:** `chip` — `'green'` or `'red'`

**Behavior:**

1. Determine how many amplifier extensions are present for the chip (2-amp or 4-amp mode).
2. Concatenate raw amp arrays into a single display image:
   - 2-amp green: `flipud(concat(AMP1, AMP2, axis=1))`
   - 2-amp red: `concat(AMP1, AMP2, axis=1)`
   - 4-amp: bottom=`concat(AMP1, AMP2, axis=1)`, top=`concat(AMP3, AMP4, axis=1)`, then `concat(bottom, top, axis=0)`
3. Check if pixel values need division by 2^16 (v2 threshold: `median > 200 * 2^16`).
4. Plot with matplotlib:
   - `plt.imshow(image, cmap='viridis', origin='lower', vmin=percentile(1), vmax=percentile(99.5))`
   - Title: `'L0 - {Chip} CCD: {ObsID} - {name}'` (fontsize=14)
   - X-label: `'Column (pixel number)'` (fontsize=14)
   - Y-label: `'Row (pixel number)'` (fontsize=14)
   - Colorbar: `shrink=0.95`, label=`'ADU'` (or `'ADU / 2^16'` if scaled), fontsize=14, tick labelsize=12
   - `plt.grid(False)`
   - Timestamp annotation (bottom right, gray, fontsize=8): `'KPF QLP: {YYYY-MM-DD HH:MM:SS} UT'`
   - Figure size: 10x8 inches, `tight_layout=True`
   - Save at 600 DPI, `facecolor='w'`
5. No read noise annotations. Those metrics are computed by ImageAssembly and stored in L1 headers; they belong on PlotL1.

**Output:**
- Returns a `matplotlib.Figure`
- If `output_dir` is set, saves to `{output_dir}/{obs_id}_L0_stitched_image_{chip}_zoomable.png`

### Method: `all()`

Convenience method. Calls `stitched_image()` for each chip present in the L0 object. Returns a dict of `{name: Figure}`.

## Standalone script

File: `scripts/qlp.py`

```
python scripts/qlp.py --obs_id KP.20230724.48905.30 --level L0 --config configs/kpf_drp_science.toml
```

- Parses obs_id to determine datecode and locate the L0 FITS file.
- Loads KPF0 from FITS.
- Creates PlotL0, calls `all()`.
- Output directory: `{output_dir}/{datecode}/{obs_id}/L0/`

The `--level` flag supports future expansion to L1, L2, L4.

## Testing

File: `tests/test_qlp_l0.py`

- Load a test L0 FITS file.
- Create PlotL0, call `stitched_image('green')` and `stitched_image('red')`.
- Assert: returns a matplotlib Figure.
- Assert: figure has expected title format.
- Assert: figure has a colorbar.
- Assert: image data shape matches expected CCD dimensions.
- No pixel-level assertions.

## Output format

- PNG, 600 DPI, white facecolor
- Filename convention: `{obs_id}_L0_stitched_image_{chip}_zoomable.png`
- Directory structure: `{output_dir}/{datecode}/{obs_id}/L0/`

## What this spec does NOT cover

- PlotL1 (v2 "2D" level plots, including read noise annotations)
- PlotL2 (extracted spectra plots)
- PlotL4 (RV/CCF plots)
- QC framework
- Exposure meter, guider, CaHK plots (these are L0-adjacent but separate)
- Recipe integration
- Time-series / nightly summary plots
