"""
KPF masters construction recipe.

Builds the nightly calibration master products for a single datecode from its
raw L0 frames: master bias and master dark (L1), and master wavelength solution
(L2) from ThAr exposures. Flat masters are scaffolded but not yet implemented.
Each master is stacked by the corresponding module under `kpfpipe.modules.masters`
and written to the output data root via the pipeline path helpers.
"""

import logging
import os

from kpfpipe.modules.masters import WLS, Bias, Dark
from kpfpipe.utils.io import FileHandler, kpf_filepath
from kpfpipe.utils.kpf_utils import get_obs_id

# Explicit name: the CLI execs recipes with __name__ == "recipe", so __name__
# would not identify this module in the log (style guide section 6).
logger = logging.getLogger("kpfpipe.recipe.masters")


def main(config, args):
    logger.info("entering kpf_drp_masters pipeline")

    if not args.datecode:
        raise SystemExit(
            "Error: --datecode is required for the masters recipe (e.g. -d 20240405)"
        )

    datecode = args.datecode
    logger.info("building masters for %s", datecode)

    data_dirs = config.get_params(["DATA_DIRS"])
    data_root_in = data_dirs["KPF_DATA_INPUT"]
    data_root_masters = data_dirs["KPF_MASTERS_OUTPUT"]

    l0_dir = os.path.join(data_root_in, "L0", datecode)
    if not os.path.isdir(l0_dir):
        raise SystemExit(f"L0 data directory not found: {l0_dir}")

    # Scan the night's L0 headers once; the handler carries the mini database
    # across the per-cal-type build_calibration_stacks calls below. cache="r"
    # reuses the on-disk mini-database CSV when present.
    file_handler = FileHandler(data_dirs)
    file_handler.build_mini_database(datecode, cache="r")

    # Stack the bias frames into a master bias used to remove the detector
    # offset from every science and calibration frame.
    for files in file_handler.build_calibration_stacks(
        "bias",
        min_stack_size=config.get_params(["BIAS"]).get("min_stack_size"),
        groupby="time_of_day",
    ):
        bias_path = kpf_filepath(
            get_obs_id(files[0]), "L1", data_root=data_root_masters, master="bias"
        )
        logger.info("stacking %d bias frames -> %s", len(files), bias_path)
        bias = Bias(files, config)
        bias.make_master_l1(master_path=bias_path)

    # Stack the dark frames into a master dark used to remove dark current.
    # Runs after the master bias so CalibrationAssociation can subtract that
    # bias from each dark frame (via _process_frame) before stacking.
    for files in file_handler.build_calibration_stacks(
        "dark",
        min_stack_size=config.get_params(["DARK"]).get("min_stack_size"),
        groupby="obs_night",
    ):
        dark_path = kpf_filepath(
            get_obs_id(files[0]), "L1", data_root=data_root_masters, master="dark"
        )
        logger.info("stacking %d dark frames -> %s", len(files), dark_path)
        dark = Dark(files, config)
        dark.make_master_l1(master_path=dark_path)

    # master flat (not yet implemented)
    # for files in file_handler.build_calibration_stacks('flat'):
    #    flat_path = kpf_filepath(get_obs_id(files[0]), 'L1',
    #                               data_root=data_root_masters, master='flat')
    #    flat = Flat(files, config)
    #    flat.make_master_l1(master_path=flat_path)

    # Stack the ThAr exposures into a master wavelength solution, since the
    # emission-line spectrum anchors the per-order wavelength calibration.
    for files in file_handler.build_calibration_stacks(
        "thar",
        min_stack_size=config.get_params(["WLS"]).get("min_stack_size"),
        groupby="time_of_day",
    ):
        obs_id = get_obs_id(files[0])
        wls_path = kpf_filepath(
            obs_id, "L2", data_root=data_root_masters, master="thar"
        )
        logger.info("building WLS from %d ThAr frames -> %s", len(files), wls_path)
        wls = WLS(files, config)
        wls.make_master_l2(master_path=wls_path)

    logger.info("exiting kpf_drp_masters pipeline")


if __name__ == "__main__":
    main()
