# KPF-Pipeline

Data Reduction Pipeline for the Keck Planet Finder spectrograph.

This branch (`kpf-next`) is the vNext rebuild: a simple, explicit,
file-in/file-out pipeline intended to make calibration logic and RV regression
work easy to inspect. The immediate basic path is:

```text
L0 raw FITS -> bias/dark/ThAr masters -> L2 extracted spectra -> L4 RVs
```

Flat files should be collected with the smoke data set, but flat correction is
currently disabled because master-flat construction and flat division are still
scaffolded.

## Environment

Create the pinned conda environment and install the package in editable mode:

```bash
conda env create -f environment.yml
conda activate kpfpipe
pip install -e .
```

All test and pipeline commands should run from the repo root in this
environment. The `Makefile` test targets call `conda run -n kpfpipe ...`.

## Local Data Layout

The default configs use this base directory:

```text
/data/kpf
```

Expected layout for a first smoke run:

```text
/data/kpf/L0/20240405/*.fits
/data/kpf/masters-root/masters/20240405/*.fits
/data/kpf/science-root/L2/20240405/*.fits
/data/kpf/science-root/L4/20240405/*.fits
/data/kpf/science-root/QLP/20240405/<obs_id>/*/*.png
```

Minimum L0 files for rebuilding masters locally:

- At least 5 bias frames with `OBJECT=autocal-bias`
- At least 5 dark frames with `OBJECT=autocal-dark`
- At least 5 ThAr frames with `OBJECT=autocal-thar-all-*`
- One science exposure, initially `KP.20240405.40113.57.fits`
- Flats with `OBJECT=autocal-flat-all`, if available, for future flat work

The pipeline builds a mini CSV database from the primary headers in each
`L0/<datecode>` directory. FITS data under `tests/testdata/` is intentionally
gitignored and should not be committed.

## Smoke Commands

Run the fast synthetic/unit suite first:

```bash
make test-fast
```

Build a single night's calibration masters (in-process, one recipe run):

```bash
kpfpipe run --masters -d 20240405 \
  --kpf_data_input /data/kpf \
  --kpf_masters_output /data/kpf/masters-root
```

Reduce the canonical smoke science exposure (in-process):

```bash
kpfpipe run --science -o KP.20240405.40113.57 \
  --kpf_data_input /data/kpf \
  --kpf_masters_output /data/kpf/masters-root \
  --kpf_science_output /data/kpf/science-root
```

The masters recipe writes master bias/dark L1 products and a master ThAr WLS L2
product. The science recipe writes L2/L4 FITS products and quicklook PNGs.

`--masters`/`--science` set the recipe **and** a default config; pass `-c` to run
that recipe against a custom config (e.g. `kpfpipe run --science -c my.toml -o …`).

`kpfpipe` is a dispatcher: `kpfpipe run` reduces one recipe on one unit
(above), while `kpfpipe masters` / `kpfpipe science` fan out a batch — one
`run` subprocess per unit, each with its own log and exit code:

```bash
kpfpipe masters --dates 20240405 20240712 --kpf_data_input /data/kpf ...
kpfpipe masters --dates nights.txt   # or a file listing one datecode per line
kpfpipe science --obs_id_list KP.20240405.40113.57 KP.20240405.40237.36 ...
```

## Quicklook Full-Resolution Images

L0 stitched-image and L1 assembled-image quicklooks support an opt-in
`full_res=True` mode on the plot object constructor, individual image method, or
`run(...)`. This writes one PNG pixel per CCD pixel with the same colormap and
scaling as the default view. It is off by default because native detector PNGs
are much larger and slower to write than the standard downsampled quicklooks.

## Current Limitations

- Python is pinned to `3.14.3` in `environment.yml` and `pyproject.toml`.
- Flat correction is not part of the basic runnable path yet.
- The WMKO archive hook exists as a local no-op placeholder in
  `file_io_hooks.py`; production write-hook integration is still follow-on work.
- One-shot, realtime, and multi-night reprocessing entry points are not yet the
  first bring-up target.
