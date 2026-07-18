"""Quicklook framework base class.

The quicklook layer renders per-level PNG plots of finished data products for
near-real-time observing feedback (WMKO DRP-RUN-01). Each ``PlotL{n}`` subclass
implements the plot methods named in its ``_PLOT_METHODS`` tuple plus a
``_has_chip`` presence test; the shared ``run()`` dispatches the requested
plot(s) across the GREEN/RED CCDs, saving-and-closing when ``output_dir`` is set
and returning the open figures otherwise.

Unlike the Diagnostics -> QC -> Checkpoints stages, quicklook is pure
visualization: it never writes to product headers or science extensions.
"""

import logging
import os
from datetime import UTC, datetime

import matplotlib.pyplot as plt

from kpfpipe.quality_control.quicklook._save_png import save_png

logger = logging.getLogger(__name__)

_CHIPS = ("green", "red")
_DPI = 200


class Plot:
    """Base runner for per-level quicklook plots.

    Parameters
    ----------
    kpf_obj : KPFDataModel
        Finished data product to visualize (read-only).
    output_dir : str or None
        Directory to save PNG files. None returns the Figures without saving.
    obs_id : str or None
        Observation ID for titles/filenames. None falls back to
        ``kpf_obj.obs_id`` (populated on every construction path).
    """

    LEVEL = None  # Subclasses set the level tag ("L0", "L1", "L2", "L4").
    _PLOT_METHODS = ()  # Subclasses list their plot-method names, in order.

    def __init__(self, kpf_obj, output_dir=None, obs_id=None):
        self.kpf_obj = kpf_obj
        self.output_dir = output_dir
        self.obs_id = obs_id or kpf_obj.obs_id
        self.name = kpf_obj.headers["PRIMARY"]["OBJECT"]

    def _has_chip(self, chip):
        """True if ``chip`` ('green'/'red') holds plottable data. Subclass hook."""
        raise NotImplementedError

    def _timestamp(self):
        """Current UTC time as the standard QLP annotation string."""
        return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    def _decorate_and_save(self, fig, plot_name, chip):
        """Add the standard QLP timestamp and save if ``output_dir`` is set."""
        fig.text(
            0.99,
            0.005,
            f"KPF QLP: {self._timestamp()} UT",
            fontsize=8,
            color="darkgray",
            ha="right",
            va="bottom",
        )
        if self.output_dir is not None:
            prefix = f"{self.obs_id}_" if self.obs_id else ""
            path = os.path.join(
                self.output_dir,
                f"{prefix}{self.LEVEL}_{plot_name}_{chip.lower()}_zoomable.png",
            )
            save_png(fig, path, dpi=_DPI, compress_level=6)
        return fig

    def _resolve_plot_names(self, which):
        """Map ``which`` ('all' or a single method name) to the methods to run."""
        if which == "all":
            return self._PLOT_METHODS
        if which in self._PLOT_METHODS:
            return (which,)
        raise ValueError(
            f"unknown plot {which!r}; expected 'all' or one of {self._PLOT_METHODS}"
        )

    def run(self, which, **plot_kwargs):
        """Generate the requested plot(s) for every chip that has data.

        Saved-and-closed when ``output_dir`` is set, returned open when it is
        None. ``plot_kwargs`` forward to each plot method (e.g. ``full_res`` for
        the L0/L1 image plots). A plot method that raises is logged at ERROR
        (naming the plot and chip) and its exception propagates unchanged --
        fail-fast; this stage records rather than swallows.

        Parameters
        ----------
        which : str
            'all' to run every plot in ``_PLOT_METHODS``, or a single method name.

        Returns
        -------
        dict
            Maps ``{method_name}_{chip}`` to its matplotlib.Figure.

        Raises
        ------
        ValueError
            If ``which`` is neither 'all' nor a known plot method name.
        """
        names = self._resolve_plot_names(which)
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

        figures = {}
        for chip in _CHIPS:
            if not self._has_chip(chip):
                logger.debug(
                    "%s quicklook: no data for %s CCD, skipping", self.LEVEL, chip
                )
                continue
            for name in names:
                try:
                    fig = getattr(self, name)(chip, **plot_kwargs)
                    if fig is None:
                        continue
                    figures[f"{name}_{chip}"] = fig
                    # Close only saved figures (open ones are returned for display).
                    if self.output_dir is not None:
                        plt.close(fig)
                except Exception as e:
                    logger.error(
                        "%s plot %r on %s raised: %s", self.LEVEL, name, chip, e
                    )
                    raise
        return figures
