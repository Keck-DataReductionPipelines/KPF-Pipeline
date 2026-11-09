"""kpfpipe.utils.target: parsing, comparing and reconciling target names."""

import re

import pytest

from kpfpipe.utils.target import (
    _NUMERIC_CATALOGS,
    normalize_name,
    parse_target,
    reconcile_gaiaid,
    reconcile_object,
    resolve_identity,
    same_target,
)

_G19 = "2452378776434477184"


# ------------------------------------------------------------------
# parse_target: catalog patterns, letters, Gaia fields
# ------------------------------------------------------------------


class TestParseTarget:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("57555", ("HD", "HD 57555", "")),
            ("hd21019a", ("HD", "HD 21019A", "")),
            ("HD 189733b", ("HD", "HD 189733", "b")),
            (
                "Gaia_EDR3_2077618037775709568",
                ("Gaia", "Gaia DR3 2077618037775709568", ""),
            ),
            ("hr3662", ("HR", "HR 3662", "")),
            ("GJ12", ("GJ", "GJ 12", "")),
            ("GJ 105.5", ("GJ", "GJ 105.5", "")),
            ("GJ 0105.5", ("GJ", "GJ 105.5", "")),
            ("GJ 0105", ("GJ", "GJ 105", "")),
            ("GJ 9066.1", ("GJ", "GJ 9066.1", "")),
            # Uncatalogued coordinate names keep every digit.
            ("SDSS J123456.89-012345.6", ("", "SDSS J123456.89-012345.6", "")),
            ("2MASS J18365633+3847012", ("", "2MASS J18365633+3847012", "")),
            ("GL699", ("GJ", "GJ 699", "")),
            ("Gliese12", ("GJ", "GJ 12", "")),
            ("gliese 12", ("GJ", "GJ 12", "")),
            ("Gl 15A", ("GJ", "GJ 15A", "")),
            ("G 32-5", ("", "G 32-5", "")),
            ("HR 0001", ("HR", "HR 1", "")),
            ("KOI-0984", ("KOI", "KOI-984", "")),
            ("TOI-04018", ("TOI", "TOI-4018", "")),
            ("T004018", ("TOI", "TOI-4018", "")),
            ("T4018", ("TOI", "TOI-4018", "")),
            ("K00148", ("KOI", "KOI-148", "")),
            ("CK01442", ("KOI", "KOI-1442", "")),
            ("K218", ("K2", "K2-18", "")),
            ("MASCARA1", ("MASCARA", "MASCARA-1", "")),
            ("HATS-2", ("HATS", "HATS-2", "")),
            ("LHS 1140 b", ("LHS", "LHS 1140", "b")),
            ("V1298Tau", ("GCVS", "V1298 Tau", "")),
            ("V* V488 Per", ("GCVS", "V488 Per", "")),
            ("EPIC212036875", ("EPIC", "EPIC 212036875", "")),
            ("BD+04_4988", ("BD", "BD+04 4988", "")),
            ("CD-38 13209", ("CD", "CD-38 13209", "")),
            ("CPD-46 127", ("CPD", "CPD-46 127", "")),
            ("BD+37 4734A", ("BD", "BD+37 4734A", "")),
            ("TYC-4209-1322-1", ("TYC", "TYC 4209-1322-1", "")),
            ("ADS16402A", ("ADS", "ADS 16402A", "")),
            ("WD1337+705", ("WD", "WD 1337+705", "")),
            ("WD 0046+051", ("WD", "WD 0046+051", "")),
            ("Qatar-1", ("Qatar", "Qatar 1", "")),
            ("KELT-2A", ("KELT", "KELT-2A", "")),
            ("XO-2N", ("XO", "XO-2N", "")),
            ("TRES1", ("TrES", "TrES-1", "")),
            (f"Gaia DR3 {_G19}", ("Gaia", f"Gaia DR3 {_G19}", "")),
            ("HAT-P-4B", ("HAT-P", "HAT-P-4B", "")),
            ("HATP7b", ("HAT-P", "HAT-P-7", "b")),
            ("tau Cet", ("", "tau Cet", "")),
            ("BL_Lac", ("", "BL Lac", "")),
        ],
    )
    def test_canonical(self, name, expected):
        parsed = parse_target(name)
        assert (parsed.catalog, parsed.name, parsed.planet) == expected

    def test_component(self):
        assert parse_target("hd21019a").component == "A"
        assert parse_target("HD 189733b").component == ""

    @pytest.mark.parametrize(
        "name, release",
        [(f"Gaia DR3 {_G19}", "DR3"), (f"dr2_{_G19}", "DR2"), (_G19, "")],
    )
    def test_gaia_fields(self, name, release):
        parsed = parse_target(name)
        assert (parsed.gaia_release, parsed.gaia_source_id) == (release, _G19)

    def test_numeric_catalogs(self):
        # Oracle: pins the csv "numeric" column.
        assert _NUMERIC_CATALOGS == {"Gaia", "TIC", "KIC", "EPIC"}

    def test_name_matching_several_rows_raises(self, monkeypatch):
        # Row order never picks a winner: overlapping patterns are a csv bug.
        rows = [
            ("HD", re.compile(r"(\d+)"), "HD {}"),
            ("HIP", re.compile(r"(\d+)"), "HIP {}"),
        ]
        monkeypatch.setattr("kpfpipe.utils.target._ALIASES", rows)
        with pytest.raises(ValueError, match="matches several"):
            parse_target("10700")


# ------------------------------------------------------------------
# normalize_name: SIMBAD markers and separators
# ------------------------------------------------------------------


class TestNormalizeName:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("*  61 Cyg B", "61 Cyg B"),
            ("V* V1298 Tau", "V1298 Tau"),
            ("NAME Barnard's star", "Barnard's star"),
            ("BL_Lac", "BL Lac"),
            ("  HD  10700 ", "HD 10700"),
        ],
    )
    def test_normalize(self, raw, expected):
        assert normalize_name(raw) == expected


# ------------------------------------------------------------------
# same_target: name equivalence
# ------------------------------------------------------------------


class TestSameTarget:
    @pytest.mark.parametrize(
        "a, b",
        [
            ("V1298Tau", "V1298 Tau"),
            ("V488Per", "V488 Per"),
            ("BL_Lac", "BL Lac"),
            ("TW_Hya", "tw hya"),
            ("HE1033+0059", "HE 1033+0059"),
            ("HE1032-1655", "HE 1032-1655"),
            ("WD1337+705", "WD 1337+705"),
            ("Feige67", "Feige 67"),
            ("BD+04_4988", "BD+04 4988"),
            ("ADS16402A", "ADS 16402 A"),
            ("Qatar-1", "Qatar 1"),
            ("Gl 15A", "GJ 15 A"),
        ],
    )
    def test_equal(self, a, b):
        assert same_target(a, b)

    @pytest.mark.parametrize(
        "a, b",
        [
            ("BD+03 2870", "BD-03 2870"),
            ("GJ 1055", "GJ 105 5"),
            ("GJ 105.5", "GJ 1055"),
            ("HD 10700", "HD 10701"),
        ],
    )
    def test_unequal(self, a, b):
        assert not same_target(a, b)


# ------------------------------------------------------------------
# reconcile_object: OBJECT from OBJECT and TARGNAME
# ------------------------------------------------------------------


class TestReconcileObject:
    @staticmethod
    def _run(obj, targname, gaiaid=f"DR3 {_G19}"):
        warnings = []
        return reconcile_object(obj, targname, gaiaid, warnings.append), warnings

    @pytest.mark.parametrize(
        "obj, targname, expected",
        [
            ("SoCal", "Sun", "SoCal"),
            ("Io", "Io Eclipse", "Io"),
            ("hd10700", "HD 10700", "HD 10700"),
            ("WASP-173", "WASP-173B", "WASP-173B"),
            ("TOI", "TOI-969", "TOI-969"),
            ("HD 10700", "HD 10701", "Unknown"),
        ],
    )
    def test_object(self, obj, targname, expected):
        assert self._run(obj, targname)[0] == expected

    def test_unknown_warns_reason(self):
        _, warnings = self._run("HD 10700", "HD 10701")
        assert warnings == [
            "OBJECT 'HD 10700' / TARGNAME 'HD 10701': the cards name different "
            "targets ('HD 10700' vs 'HD 10701'); OBJECT is Unknown; AstroQuery "
            f"resolves OBJECT and astrometry from GAIAID 'Gaia DR3 {_G19}'"
        ]

    def test_blank_card_warns_written_name(self):
        assert self._run("hd189733b", "") == (
            "HD 189733",
            [
                "OBJECT 'hd189733b' names planet 'b'; host star 'HD 189733' (a "
                "trailing letter can denote a binary component or a planet)",
                "TARGNAME blank; OBJECT is 'HD 189733'",
            ],
        )

    def test_blank_card_bare_number_warns_once(self):
        _, warnings = self._run("343600706696532", "")
        assert len(warnings) == 1
        assert "a bare number in no known catalog; OBJECT is Unknown" in warnings[0]

    def test_letter_on_both_cards_warns_once(self):
        _, warnings = self._run("HD 189733b", "HD189733b")
        assert warnings == [
            "OBJECT 'HD 189733b' / TARGNAME 'HD189733b' names planet 'b'; host star "
            "'HD 189733' (a trailing letter can denote a binary component or a planet)"
        ]

    def test_uncatalogued_letter_case_mismatch_warns(self):
        result, warnings = self._run("Ross 128 b", "Ross 128 B")
        assert result == "Ross 128 b"
        assert warnings == [
            "OBJECT 'Ross 128 b' / TARGNAME 'Ross 128 B' differ in the case of a "
            "trailing letter, which can denote a planet or a binary component; "
            "OBJECT is 'Ross 128 b'"
        ]

    def test_uncatalogued_same_letter_case_silent(self):
        assert self._run("Ross 128 B", "Ross128B") == ("Ross 128 B", [])

    def test_unknown_without_gaiaid_raises(self):
        with pytest.raises(ValueError, match="GAIAID blank"):
            self._run("HD 10700", "HD 10701", gaiaid="")

    def test_planet_and_component_raises(self):
        with pytest.raises(ValueError, match="planet letter with a component"):
            self._run("HD 189733b", "HD 189733B")


# ------------------------------------------------------------------
# reconcile_gaiaid: OBJECT vs GAIAID
# ------------------------------------------------------------------


class TestReconcileGaiaid:
    def test_release_lent_to_object(self):
        warnings = []
        result = reconcile_gaiaid(f"Gaia {_G19}", f"DR3 {_G19}", warnings.append)
        assert result == (f"Gaia DR3 {_G19}", f"Gaia DR3 {_G19}")
        assert warnings == [
            f"OBJECT 'Gaia {_G19}' takes the GAIAID release; OBJECT is "
            f"'Gaia DR3 {_G19}'"
        ]

    def test_non_gaia_object_keeps_gaiaid(self):
        warnings = []
        result = reconcile_gaiaid("HD 10700", f"DR3 {_G19}", warnings.append)
        assert result == ("HD 10700", f"Gaia DR3 {_G19}")
        assert warnings == []

    @pytest.mark.parametrize("gaiaid", ["123", "1234567890123456"])
    def test_malformed_raises(self, gaiaid):
        with pytest.raises(ValueError, match="a shorter id needs its release"):
            reconcile_gaiaid("HD 10700", gaiaid, [].append)


# ------------------------------------------------------------------
# resolve_identity: OBJECT and ALIASES from SIMBAD
# ------------------------------------------------------------------


class TestResolveIdentity:
    # SIMBAD identifiers of TOI-2473 = HD 287319, deliberately not in preference order.
    _IDS = [
        parse_target(i)
        for i in (
            "TIC 399868187",
            "Gaia DR3 3291551614399361152",
            "BD+10 678",
            "HD 287319",
            "HIP 1",
            "TOI-2473",
        )
    ]
    _GAIA = "Gaia DR3 3291551614399361152"

    def _resolve(self, obj, ids=None, gaiaid=None):
        ids = self._IDS if ids is None else ids
        return resolve_identity(obj, gaiaid or self._GAIA, self._GAIA, ids, "")

    def test_non_numeric_object_kept(self):
        obj, aliases = self._resolve("HD 287319")
        assert obj == "HD 287319"
        # Catalog preference (csv row order), not SIMBAD's order.
        assert aliases == (
            f"HD 287319; HIP 1; TOI-2473; BD+10 678; TIC 399868187; {self._GAIA}"
        )

    def test_independent_of_simbad_order(self):
        for obj in ("HD 287319", "Unknown", "TIC 399868187"):
            assert self._resolve(obj) == self._resolve(obj, self._IDS[::-1])

    @pytest.mark.parametrize(
        "obj, expected",
        [
            ("TIC 399868187", "TOI-2473"),  # TIC -> TOI
            ("Unknown", "HD 287319"),  # resolved from GAIAID -> HD
            (_GAIA, "HD 287319"),
        ],
    )
    def test_numeric_object_takes_successor(self, obj, expected):
        assert self._resolve(obj)[0] == expected

    def test_successor_fallbacks(self):
        no_hd = [i for i in self._IDS if i.catalog != "HD"]
        assert self._resolve("Unknown", no_hd)[0] == "HIP 1"
        # No TOI: the most preferred catalog (HD), not SIMBAD's first (BD).
        no_toi = [i for i in self._IDS if i.catalog != "TOI"]
        assert self._resolve("TIC 399868187", no_toi)[0] == "HD 287319"

    @pytest.mark.parametrize(
        "obj, names, expected",
        [
            ("KIC 1", ["KOI-1", "Kepler-1"], "Kepler-1"),
            ("KIC 1", ["KOI-1"], "KOI-1"),
            ("EPIC 1", ["K2-1"], "K2-1"),
        ],
    )
    def test_mission_successors(self, obj, names, expected):
        # HD 1 is the most preferred catalog: only the successor rule picks past it.
        ids = [parse_target(n) for n in (obj, "HD 1", *names)]
        assert resolve_identity(obj, "", "", ids, "")[0] == expected

    def test_object_takes_simbad_spelling(self):
        ids = [parse_target("V* V1298 Tau"), parse_target(self._GAIA)]
        assert self._resolve("V1298Tau", ids)[0] == "V1298 Tau"

    def test_bare_gaia_object_takes_designation(self):
        assert self._resolve(f"Gaia {self._GAIA.split()[-1]}", [])[0] == self._GAIA

    def test_object_missing_from_simbad_raises(self):
        with pytest.raises(ValueError, match="not an identifier of SIMBAD's match"):
            self._resolve("HD 10700")

    def test_other_gaia_source_raises(self):
        with pytest.raises(ValueError, match="name different stars"):
            self._resolve("TOI-2473", gaiaid="Gaia DR3 1111111111111111111")

    def test_bare_gaiaid_matches_simbad_any_release(self):
        bare = f"Gaia {self._GAIA.split()[-1]}"
        assert self._resolve("TOI-2473", gaiaid=bare)[0] == "TOI-2473"

    def test_bare_gaiaid_other_source_raises_with_release(self):
        with pytest.raises(ValueError, match=f"is {self._GAIA}\\. L0 header"):
            self._resolve("TOI-2473", gaiaid="Gaia 1111111111111111111")
