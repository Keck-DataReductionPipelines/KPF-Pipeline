# Reference data

## Wavelength convention

**All wavelengths in this directory are vacuum wavelengths, in Ångström (Å).**

The pipeline assumes vacuum wavelengths everywhere and performs **no on-the-fly
air↔vacuum conversion**. Any new reference file added here must be in vacuum.
(The `air_to_vac` helper in `kpfpipe/utils/astro.py` is retained for ad hoc use
but is not called in the standard data flow.)

### Wavelength-bearing files

| File | Contents | Frame |
|------|----------|-------|
| `thar_line_list.csv` | ThAr lamp line wavelengths for the WLS | vacuum |
| `rough_wls/rough_wls_<datecode>.csv` | Per-order Legendre coeffs seeding the WLS line search, one per stable stretch of an era (chosen like the order trace) | vacuum |
| `line_masks/stellar_masks/*_espresso.txt` | ESPRESSO stellar CCF masks (wavelength, weight), per spectral type | vacuum |
| `line_masks/line_mask_lookup.csv` | TEFF → stellar-mask lookup (no wavelengths) | — |
| `EEM_dwarf_UBVIJHK_colors_Teff.txt` | Dwarf colour → TEFF sequence (no wavelengths) | — |

### Provenance / verification

- `thar_line_list.csv` and the retired `rough_wls_fallback.csv` derive from the
  legacy KPF ThAr solution, whose line list is vacuum (legacy
  `known_wavelengths_vac`); confirmed by sub-mÅ agreement with that source.
- `rough_wls/rough_wls_<datecode>.csv` are that fallback (an Era-2 solution)
  refit at degree 6 after a per-chip quadratic pixel shift measured from ThAr
  lines: Era 2 (`20240224`) by line fits to 20240613, Era 1 (`20221207`) by
  cross-correlating five Era-1 nights (2022-12 to 2024-01) against 20240613.
  Eras 2.6 (`20250102`), 3.0 (`20250424`) and 4.0 (`20251030`) are copies of
  the Era-2 file: the fallback served them, so their shifts are small, but they
  are not yet measured.
- `EEM_dwarf_UBVIJHK_colors_Teff.txt` is "A Modern Mean Dwarf Stellar Color and
  Effective Temperature Sequence" (Pecaut & Mamajek 2013, ApJS 208, 9), version
  2022.04.16, from <https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt>.
  Held **verbatim as distributed** — preamble, notes footer and all — so it can be
  re-fetched and diffed against upstream. `kpfpipe.utils.astro.color_to_teff` is its
  only reader and does all parsing at load time; `G-J` is not a tabulated column and
  is formed there as `M_G - M_J`.
- The `*_espresso.txt` stellar masks are distributed by ESPRESSO in **air** and
  were converted to vacuum in-place (wavelength column only; weights unchanged)
  via `air_to_vac`. Verified against vacuum stellar-line wavelengths
  (Mg b, Fe I) to ~0.01 Å.
