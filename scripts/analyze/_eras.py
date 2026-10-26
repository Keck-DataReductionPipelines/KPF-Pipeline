"""Science-era lookup shared by the ``kpfpipe analyze`` scripts."""

import kpfpipe


def load_science_eras():
    """``[(era, start, end)]`` for the eras that are not engineering/service time."""
    eras = kpfpipe.INSTRUMENT_ERAS
    engineering = eras["Comments"].fillna("").str.lower().str.contains("engineering")
    return [
        (str(row.INSTERA), row.UT_start_date, row.UT_end_date)
        for row in eras[~engineering].itertuples()
    ]


def era_of(obs_time, eras):
    """The era covering ``obs_time``; ``None`` outside the science eras.

    Resolved at the frame's own UT time, not its datecode: a boundary can split
    a UT day (2024-02-23 is era 1.5 until 12:00 and era 2.0 after).
    """
    for era, start, end in eras:
        if start <= obs_time <= end:
            return era
    return None
