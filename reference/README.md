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
| `rough_wls_fallback.csv` | Per-order Legendre coeffs seeding the WLS line search | vacuum |
| `line_masks/stellar_masks/*_espresso.txt` | ESPRESSO stellar CCF masks (wavelength, weight), per spectral type | vacuum |
| `line_masks/line_mask_lookup.csv` | TEFF → stellar-mask lookup (no wavelengths) | — |
| `EEM_dwarf_UBVIJHK_colors_Teff.txt` | Dwarf colour → TEFF sequence (no wavelengths) | — |

### Provenance / verification

- `thar_line_list.csv` and `rough_wls_fallback.csv` derive from the legacy KPF
  ThAr solution, whose line list is vacuum (legacy
  `known_wavelengths_vac`); confirmed by sub-mÅ agreement with that source.
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
