"""Profile ``OrderTrace.make_master`` (order traces from one master flat).

Traces both CCDs of the bundled master flat: threshold the illuminated pixels
and curate them into clusters, phase the fiber pattern on the CAL orderlets, fit
every centerline, and measure the apertures. Consumes one completed L1 master
rather than a stack of L0 frames, so the time is trace measurement, not I/O.

Run with ``make profile-order_trace`` or
``python tests/profiling/profile_order_trace.py``. Requires real frames in
``tests/testdata`` (skips cleanly otherwise).
"""

try:  # importable via `-m tests.profiling.profile_*` or runnable as a script
    from tests.profiling import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

import kpfpipe.modules.masters.order_trace as m_order_trace


def run():
    def setup():
        return m_order_trace.OrderTrace(P.masters_flat_file())

    P.run_profile(
        title="Order trace (OrderTrace.make_master)",
        report_name="order_trace",
        setup=setup,
        call=lambda mod: mod.make_master(),
        candidate_modules=[m_order_trace],
    )


if __name__ == "__main__":
    run()
