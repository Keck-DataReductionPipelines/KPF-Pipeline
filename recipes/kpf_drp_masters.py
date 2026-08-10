"""
KPF masters construction recipe.

Builds the nightly calibration master products for a single datecode from its
raw L0 frames: master bias, master dark, and master flat (L1), and master
wavelength solution (L2) from ThAr exposures. Each master is stacked by the
corresponding module under `kpfpipe.modules.masters` and written to the output
data root via the pipeline path helpers.
"""

import logging
import os
import time

from kpfpipe.modules.masters import WLS, Bias, Dark, Flat, OrderTrace
from kpfpipe.utils.io import FileHandler, kpf_directory, kpf_filepath
from kpfpipe.utils.kpf import get_obs_id
from recipes._logging import masters_run_summary

# Explicit name: the CLI execs recipes with __name__ == "recipe", so __name__
# would not identify this module in the log.
logger = logging.getLogger("kpfpipe.recipe.masters")


def main(config, args):
    t0 = time.monotonic()
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

    built = []  # (cal_type, path, n_frames) per master stacked, for the run summary

    # Stack the bias frames into a master bias used to remove the detector
    # offset from every science and calibration frame.
    for files in file_handler.build_calibration_stacks(
        "bias",
        min_stack_size=config.get_params(["BIAS"])["min_stack_size"],
        max_stack_size=config.get_params(["BIAS"])["max_stack_size"],
        groupby="time_of_day",
    ):
        bias_path = kpf_filepath(
            get_obs_id(files[0]), "L1", data_root=data_root_masters, master="bias"
        )
        logger.info("stacking %d bias frames -> %s", len(files), bias_path)
        bias = Bias(files, config)
        bias.make_master_l1(master_path=bias_path)
        built.append(("bias", bias_path, len(files)))

    # Stack the dark frames into a master dark used to remove dark current.
    # Runs after the master bias so CalibrationAssociation can subtract that
    # bias from each dark frame (via _process_frame) before stacking.
    for files in file_handler.build_calibration_stacks(
        "dark",
        min_stack_size=config.get_params(["DARK"])["min_stack_size"],
        max_stack_size=config.get_params(["DARK"])["max_stack_size"],
        groupby="obs_night",
    ):
        dark_path = kpf_filepath(
            get_obs_id(files[0]), "L1", data_root=data_root_masters, master="dark"
        )
        logger.info("stacking %d dark frames -> %s", len(files), dark_path)
        dark = Dark(files, config)
        dark.make_master_l1(master_path=dark_path)
        built.append(("dark", dark_path, len(files)))

    # Stack the flat frames into a master flat. Runs after the master bias and
    # dark so CalibrationAssociation can subtract both from each flat frame (via
    # _process_frame) before stacking.
    for files in file_handler.build_calibration_stacks(
        "flat",
        min_stack_size=config.get_params(["FLAT"])["min_stack_size"],
        max_stack_size=config.get_params(["FLAT"])["max_stack_size"],
        groupby="time_of_day",
    ):
        flat_path = kpf_filepath(
            get_obs_id(files[0]), "L1", data_root=data_root_masters, master="flat"
        )
        logger.info("stacking %d flat frames -> %s", len(files), flat_path)
        flat = Flat(files, config)
        flat.make_master_l1(master_path=flat_path)
        built.append(("flat", flat_path, len(files)))

    # Trace the orderlets on each master flat. Runs after the flats, whose
    # geometry is the only input, and writes one CSV covering all CCDs beside
    # the flat it was measured from.
    masters_dir = kpf_directory(
        kind="masters", data_root=data_root_masters, datecode=datecode
    )
    for flat_path in [entry[1] for entry in built if entry[0] == "flat"]:
        logger.info("building order trace from master flat %s", flat_path)
        order_trace = OrderTrace(flat_path, config)
        order_trace.make_master(output_dir=masters_dir)
        built.append(("order_trace", order_trace.output_path, 1))

    # Stack the ThAr exposures into a master wavelength solution, since the
    # emission-line spectrum anchors the per-order wavelength calibration.
    for files in file_handler.build_calibration_stacks(
        "thar",
        min_stack_size=config.get_params(["WLS"])["min_stack_size"],
        max_stack_size=config.get_params(["WLS"])["max_stack_size"],
        groupby="time_of_day",
    ):
        obs_id = get_obs_id(files[0])
        wls_path = kpf_filepath(
            obs_id, "L2", data_root=data_root_masters, master="thar"
        )
        logger.info("building WLS from %d ThAr frames -> %s", len(files), wls_path)
        wls = WLS(files, config)
        wls.make_master_l2(master_path=wls_path)
        built.append(("thar", wls_path, len(files)))

    logger.info(masters_run_summary(datecode, built, time.monotonic() - t0))
    logger.info("exiting kpf_drp_masters pipeline")


if __name__ == "__main__":
    main()
