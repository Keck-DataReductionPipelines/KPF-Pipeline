# EPRV Data Standard — KPF-relevant reference

This file mirrors the portions of the **EPRV FITS Data Standard** that govern KPF data
products, condensed from the official docs:
<https://eprv-data-standard.readthedocs.io/en/develop/> (the `develop` branch — the standard
changes slowly enough that we track it rather than pin a commit; re-scrape if it has moved).
Only KPF-relevant content is reproduced; consult the source for the full standard (all
instruments, L3, complete keyword tables, the translator API).

**Authority precedence.** When requirements or design principles conflict, the order is:
**1. the WMKO technical requirements (`WMKO_REQUIREMENTS.md`) → 2. this EPRV standard →
3. the project charter (`KPF_DRP_VNEXT_CHARTER.md`) → 4. the style guide
(`KPF_DRP_VNEXT_STYLE_GUIDE.md`).** The WMKO requirements outrank this standard; below them,
KPF L2 and L4 products are EPRV-compliant by contract, so this standard wins on data
structure, extension/keyword names, units, and reference frames.

---

## 1. Purpose & data-level model

The EPRV standard defines a unified FITS format for processed echelle-spectrograph data so
that every instrument exposes the same structure, headers, units, and reference frames. New
instruments may adopt it natively; existing ones supply a *translator*. KPF produces
**L2 (extracted spectra)** and **L4 (RVs/CCFs)** as compliant products.

| Level | Meaning | Standardized? |
|---|---|---|
| **L0** | Native raw detector output | No (instrument-internal) |
| **L1** | Assembled image data (bias/overscan/etc.) | No (instrument-internal) |
| **L2** | 1-D order-by-order flux, wavelength, variance, blaze in per-trace extensions; barycentric correction **computed but not applied** to wavelengths | **Yes** |
| **L3** | Orders stitched into a single 1-D spectrum; blaze-corrected; drift **and** barycentric applied to wavelengths | Yes (KPF does not currently produce L3) |
| **L4** | Derived products: radial velocities, CCFs, activity metrics | **Yes** |

Implication for our hierarchy: `KPF0`/`KPF1` are KPF-internal (they wrap `KPFDataModel`),
while `KPF2`/`KPF4` subclass the EPRV `RV2`/`RV4` and must satisfy everything below.

---

## 2. Units & reference frames (binding)

| Quantity | Standard | Notes |
|---|---|---|
| Wavelength solution | **Vacuum** | All wavelengths are vacuum; never air |
| Wavelength units | Ångström or micron | 64-bit recommended (and required for `*_WAVE`) |
| Barycentric correction | **Solar-barycenter rest frame** | **Excludes systemic velocity** |
| Flux | Photoelectrons | |
| Time | **BJD_TDB** and UTC | |
| Exposure time | Seconds | |
| SNR | per pixel | wavelength via `EXSNRW*` keyword |
| Coordinates (RA/Dec) | Sexagesimal | |
| Epoch | Native survey epoch | e.g. 2015.5 for Gaia DR2 |
| Proper motion | mas/yr | |
| Color index | Gaia Bp − Rp | default unless specified |
| Systemic velocity / CCF velocity / RV | km/s | |

---

## 3. FITS structure conventions

- **PRIMARY header required** on every L2/L4 file (the EPRV Standard FITS Header).
- **Multiplicity**: extensions flagged multiplicity can repeat per trace — `TRACE1_*`,
  `TRACE2_*`, … `RV1`/`RV2`, `CCF1`/`CCF2`, etc.
- **Required vs optional**: `Required=True` extensions must be present *and meaningfully
  populated*; `Required=False` are optional enhancements.
- **MinBitDepth**: the standard enforces minimum precision — **`*_WAVE` and `BJD_TDB` are
  64-bit**; quality arrays are 8-bit. (rvdata 0.4.0 enforces this on serialize, upcasting any
  sub-64-bit `*_WAVE`/`BJD_TDB` and warning; KPF `*_WAVE`/`BJD_TDB` are already born-64.)
- Undefined values (e.g. unextracted orders) are **NaN**.

### Shared extensions (present at L2, L3, L4)

| Extension | Type | Required | Content |
|---|---|---|---|
| `PRIMARY` | PrimaryHDU | Yes | EPRV Standard FITS Header (no data) |
| `INSTRUMENT_HEADER` | ImageHDU | Yes | Native instrument header, carried through (no data) |
| `RECEIPT` | BinTableHDU | Yes | Processing log (see §7) |
| `DRP_CONFIG` | BinTableHDU | Yes | Pipeline config as a 2-column table (ConfigParser-style `key = value`) |
| `EXT_DESCRIPT` | BinTableHDU | Yes | Extension inventory: name + description |

---

## 4. Level 2 extensions

Order-by-order flux/wavelength/variance/blaze per trace. `TRACE1_WAVE` carries **drift
correction only** at L2; barycentric correction is provided separately (not applied).

| Extension | Type | Mult. | Req. | Content / shape |
|---|---|:--:|:--:|---|
| `ORDER_TABLE` | BinTable | | Yes | Trace-1 order wavelength extents: physical & index echelle order, start/end wavelength |
| `TRACE1_FLUX` | Image | ✓ | Yes | Order-by-order flux, shape `(NUMORDER, Mpix)` |
| `TRACE1_WAVE` | Image | ✓ | Yes | Vacuum wavelength, matches FLUX; NaN where undefined; **64-bit** |
| `TRACE1_VAR` | Image | ✓ | Yes | Variance, matches FLUX |
| `TRACE1_BLAZE` | Image | ✓ | Yes | Blaze function, matches FLUX; `BLZNORM` keyword = normalization status |
| `BARYCORR_KMS` | Image | | Yes | Barycentric correction [km/s]; scalar, per-order `(Norder,)`, or per-pixel `(Norder, Mpix)` |
| `BARYCORR_Z` | Image | | Yes | Barycentric correction as redshift z; same shape options |
| `BJD_TDB` | Image | | Yes | Photon-weighted exposure midpoint; same shape options; **64-bit** |
| `TRACE1_DRIFT` | Image | | No | Wavelength drift map (Δλ) |
| `TRACE1_QUALITY` | Image | ✓ | No | Per-pixel quality flags, matches FLUX; **8-bit**; 0 = no flag |
| `EXPMETER` | BinTable | | No | Exposure-meter timeseries: columns = wavelengths, rows = time steps |
| `TELEMETRY` | BinTable | | No | timestamp, sensor, value, units |
| `TRACE1_TELLURIC` | Image | ✓ | No | Telluric model, matches FLUX |
| `TRACE1_SKYMODEL` | Image | ✓ | No | Sky-emission model, matches FLUX |
| `ANCILLARY_SPECTRUM` | Image | ✓ | No | Supplementary spectra (e.g. Ca II H&K from a separate spectrograph) |
| `IMAGE` | Image | ✓ | No | Support images (guider, pupil, etc.) |
| `CUSTOM1_TRACE1_{FLUX,WAVE,VAR}` | Image | ✓ | No | User-defined corrected variants (`*_WAVE` 64-bit) |

`TRACE2_*`, `TRACE3_*`, … replicate the per-trace extensions for additional traces.

---

## 5. Level 4 extensions

Derived from L2; inherits the shared extensions. **RV measurements are mandatory; CCFs are
optional** (supports non-CCF methods).

| Extension | Type | Mult. | Req. | Content |
|---|---|:--:|:--:|---|
| `RV1` | BinTable | ✓ | Yes | RV measurements. **Required cols**: `BJD_TDB` (64), `RV`, `RV_ERR`, `BC_VELOCITY`, `WAVE_START` (64), `WAVE_END` (64). **Optional cols**: `PIXEL_START/END`, `ORDER_INDEX`, `ECHELLE_ORDER`, `WEIGHT`, `RESIDUAL_DRIFT`. Header: `RVMETHOD`, `RVSTART`, `RVSTEP`, `MASK`. `WAVE_START==WAVE_END` to report a central wavelength |
| `CCF1` | Image | ✓ | No | Cross-correlation functions; shape corresponds to `RV1` |
| `DIAGNOSTICS1` | BinTable | ✓ | No | Activity/CCF metrics (FWHM, BIS, …): `BJD_TDB`, metric_name, value, error |
| `CUSTOM_CCF1` | Image | ✓ | No | Alternative CCFs (different masks/methods) |
| `CUSTOM_RV1` | BinTable | ✓ | No | RVs from `CUSTOM_CCF1`; mirrors `RV1` |

L4 PRIMARY gains: `BJDTDB`, `RV`, `RVERR`, `BERV`, `RVMETHOD`, `SYSVEL`.
`RV2`/`CCF2`/… replicate per trace.

> **vNext note (reconciled against rvdata 0.4.0):** the table cell above mirrors
> the standard's *develop* readthedocs, which drift from the released package KPF pins:
> - **Barycentric column** — develop docs say `BC_VELOCITY`, but 0.4.0's machine-readable
>   `L4-RV_TABLE-columns.csv` (what its compliance checker uses) names it **`BERV`**. Our
>   code emits `BERV`; we are compliant. The develop-doc name is upstream drift, not ours.
> - **Headers** — develop docs list `RVMETHOD/RVSTART/RVSTEP/MASK` on `RVn`, but 0.4.0
>   registers `RVMETHOD/SKYRMVD/TELLRMVD` on `RVn` and puts the velocity-grid keywords
>   (`VELSTART/VELSTEP/VELNSTEP/CCFMASK`) on `CCFn`. Our code emits exactly the 0.4.0 set.
> - **Optional columns** — KPF populates the EPRV-optional `ORDER_INDEX`, `ECHELLE_ORDER`
>   (physical grating order, blue→red), and `WEIGHT` (per-order CCF-combination weight),
>   plus a KPF-custom `ORDER_ID` (chip/fiber/order name, 1-based per chip — the standard
>   permits team-added columns).
> - **Structural cards as keywords** — 0.4.0's per-extension keyword CSVs redundantly
>   register FITS *structural* cards astropy writes on its own (`XTENSION`, `EXTNAME`) as
>   if they were content keywords. The registry sanitizes these out at build so structural
>   and registered stay disjoint (see `keyword_registry._build_registry`).
> - **CTYPE on 1-D extensions** — 0.4.0 marks both `CTYPE1` *and* `CTYPE2` Required on every
>   RV/CCF/barycorr extension with identical boilerplate. The barycentric extensions
>   (`BJD_TDB`/`BARYCORR_KMS`/`BARYCORR_Z`) are 1-D per-order arrays, so only `CTYPE1`
>   (`Order-N`) applies; KPF stamps that and treats `CTYPE2` as N/A for 1-D. RV/CCF (2-D)
>   carry both. KPF treats `CTYPE` as registered content (axis meaning), not a structural
>   WCS card.
> - **KPF `CCF_VARn` extension** — the standard defines no CCF-variance product (`CCFn` is
>   value-only). KPF adds `CCF_VARn`, an `ImageHDU` per-velocity-bin CCF photon-variance cube
>   paired 1:1 with `CCFn` (same shape/numbering/per-chip access), persisting what the RV
>   photon-error step needs. Additive and team-scoped, following the `QUALITY_CONTROL`
>   precedent — it does not alter any standard-defined extension.

---

## 6. PRIMARY header keywords (KPF-relevant subset)

The full keyword table (environment/TCS/weather/etc.) is at the source. The keywords KPF's
L2/L4 products and the barycentric/RV path depend on:

| Keyword | Description | Units | Req. |
|---|---|---|:--:|
| `INSTRUME` | Instrument name (`KPF`) | | Yes |
| `DATALVL` | Data product level (`L2`/`L4`) | | Yes |
| `FILENAME` | FITS filename | | Yes |
| `DATE-OBS` | Exposure start | UTC | Yes |
| `JD_UTC` | Julian date of `DATE-OBS` | day | Yes |
| `EXPTIME` | Exposure time | s | Yes |
| `OBJECT` | Primary object name | | Yes |
| `NUMORDER` | Number of orders | | Yes |
| `NUMTRACE` | Trace iterator (object keyword count) | | Yes |
| `TRACE1…TRACE#` | Trace object type (`SCI`/`CAL`/`SKY`) | | Yes |
| `CLSRC1…CLSRC#` | Calibration source name (`LFC`/`Etalon`/`UNe`/…) | | Yes |
| `CSRC#`,`CID#` | Catalog source / designation | | Yes |
| `CRA#`,`CDEC#` | Catalog RA / Dec | sexagesimal | Yes |
| `CEQNX#`,`CEPCH#` | Catalog equinox / epoch | yr | Yes |
| `CRV#` | Catalog systemic RV | km/s | Yes |
| `CPLX#` | Catalog parallax | mas | No |
| `CPMR#`,`CPMD#` | Catalog proper motion RA / Dec | arcsec/yr | No |
| `OBSERVAT`,`TELESCOP` | Observatory / telescope | | Yes |
| `OBSLON/LAT/ALT` | Observatory location | deg/deg/m | Yes |
| `OBSGEO-X/Y/Z` | Cartesian site coordinates | m | No |
| `EXTRACT` | 1-D extraction type (`sum`/`flatrelative`/`optimal`) | | No |
| `EXSNR#`,`EXSNRW#` | Extracted SNR + its wavelength | SNR/pix, Å | No |
| `DRPTAG`,`EPRVTAG`,`VOCLASS` | DRP / EPRV-standard versions | | Yes |
| `DRPHASH` | Git commit hash | | No |
| `INSTERA` | Instrument-era tag (see §9) | | Yes |
| `FULLCOMP` | EPRV-standard compliance (`Yes`/`No`) | | Yes |
| `SUMMFLAG`,`TELFLAG`,`INSTFLAG`,`DRPFLAG` | Quality flags (`Pass`/`Fail`/`Warn`) | | Yes/No |
| `DQLVL0/1/2` | Per-level quality-check bitfields | | Yes |
| (L4) `RV`,`RVERR`,`BERV`,`BJDTDB`,`RVMETHOD`,`SYSVEL` | Summary RV keywords | km/s, day | — |

The catalog `C*#` block (systemic RV, parallax, proper motion, RA/Dec, epoch) is exactly what
the barycentric-correction path consumes.

---

## 7. Receipt & provenance

The `RECEIPT` extension is a processing log (modeled on the original KPF pipeline). Each row
records one processing event for reproducibility, and is inherited from lower to higher
levels. The standard's logged columns: **time, code release, branch name, commit hash,
function, args, status**. Adding an entry takes **three caller-supplied strings: function
name, relevant parameters (args), and status**; time/version/branch are filled automatically.

> **vNext note (reconciled on rvdata 0.4.0):** our code now calls the 3-arg
> `receipt_add_entry(function, args, status)` matching the standard — `KPFDataModel`
> overrides it (`data_models/base.py`) and every module passes the three strings (most an
> empty `args`). The earlier 2-arg form is gone; this drift is resolved.

---

## 8. KPF mapping & deviations (vNext vs the EPRV KPF translator)

KPF: Keck, 445–870 nm, R≈100k. Standardized L2/L4 filenames: `kpf_SL2_…fits`,
`kpf_SL4_…fits`.

**Trace ↔ fiber mapping** — KPF assigns trace numbers in **physical fiber order** on the slicer
(`SKY, SCI1, SCI2, SCI3, CAL`), consistent across `data_models/config/trace-map.csv`,
`detector.toml [fiber_positions]`, and the `[TRACES].fibers` list. This fiber↔trace-number
assignment is a **KPF convention**: the EPRV standard governs the per-trace extension *structure*
(`TRACE{N}_*`), not which fiber is trace N. (An earlier EPRV KPF translator doc listed a CAL-first
numbering; that was stale and does not bind vNext.)

| Trace | Object type | KPF fiber |
|:--:|:--:|---|
| 1 | SKY | Sky |
| 2 | SCI | Science slice 1 (`SCI1`) |
| 3 | SCI | Science slice 2 (`SCI2`) |
| 4 | SCI | Science slice 3 (`SCI3`) |
| 5 | CAL | Calibration |
| 6 | SCI | **Virtual fiber** (EPRV translator only, not produced by vNext) — `SCI1/2/3` resampled onto SCI2's grid, outlier-rejected and weighted |

**Known deviations from the EPRV KPF translator doc** (the doc describes the official
translator; vNext may differ by design per the charter — confirm each is intended):

1. **No Trace 6 (virtual combined science fiber).** Our `trace-map.csv` and `fibers`
   (`SKY,SCI1,SCI2,SCI3,CAL`) stop at trace 5; the standard defines a 6th combined-science
   trace. Structural gap if EPRV compliance requires it.
2. **Extraction method.** vNext default is `box` (`spectral_extraction`); the EPRV KPF doc
   states `optimal extraction`. (PRIMARY `EXTRACT` should reflect whatever we ship.)
3. **WLS polynomial order.** vNext default `polyorder_x = 6`; the doc states a **9th-order**
   Legendre fit.
4. **WLS source.** vNext fits a ThAr line list (rough WLS + `thar_line_list.csv`); the doc
   describes an **LFC-based** solution interpolated between bracketing LFC frames.
5. **Ca H&K naming.** We alias `CA_HK → ANCILLARY_SPECTRUM` (the standard's home for CaHK,
   so compliant); the KPF doc page names `CA_HK_SCI_WAVE`/`CA_HK_SCI_FLUX`. Verify the
   on-disk extension name we emit.
6. **`ANCILLARY_SPECTRUM` HDU type.** §4 defines it as an `Image`, but vNext creates it as
   an empty `BinTableHDU` placeholder (`data_models/level2.py`). Ca H&K extraction is still
   WIP and existing master/L2 products (including the truth dataset) encode it as a
   `BinTableHDU`, so flipping the model type breaks reading them back. Switch to `ImageHDU`
   when Ca H&K is built and products are regenerated.

**Instrument eras (`INSTERA`)**: the KPF era table is vendored at
[`reference/kpf_instrument_eras.csv`](reference/kpf_instrument_eras.csv) (era tag, UT
start/end, comment) — use it to stamp/validate `INSTERA` by observation date.
