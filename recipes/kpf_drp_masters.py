"""
KPF masters construction recipe.

Builds the nightly calibration master products for a single datecode from its
raw L0 frames: master bias (L1), and master wavelength solution (L2) from ThAr
exposures. Dark and flat masters are scaffolded but not yet implemented. Each
master is stacked by the corresponding module under `kpfpipe.modules.masters`
and written to the output data root via the pipeline path helpers.
"""

import os

from kpfpipe.modules.masters.bias import Bias

# from kpfpipe.modules.masters.dark import Dark
# from kpfpipe.modules.masters.flat import Flat
from kpfpipe.modules.masters.wls import WLS
from kpfpipe.utils.kpf import get_obs_id
from kpfpipe.utils.pipeline import (
    build_filepath,
    build_l0_file_lists,
    build_mini_database,
)


def main(config, args):
    print("\n\n=== entering kpf_drp_masters pipeline ===\n\n")

    if not args.datecode:
        raise SystemExit(
            "Error: --datecode is required for the masters recipe (e.g. -d 20240405)"
        )

    datecode = args.datecode

    data_dirs = config.get_params(["DATA_DIRS"])
    data_root_in = data_dirs["KPF_DATA_INPUT"]
    data_root_masters = data_dirs["KPF_MASTERS_OUTPUT"]

    l0_dir = os.path.join(data_root_in, "L0", datecode)
    if not os.path.isdir(l0_dir):
        raise SystemExit(f"L0 data directory not found: {l0_dir}")
    mini_db = build_mini_database(l0_dir)

    # Stack the bias frames into a master bias used to remove the detector
    # offset from every science and calibration frame.
    for files in build_l0_file_lists("bias", mini_db=mini_db):
        bias_path = build_filepath(
            get_obs_id(files[0]), "L1", data_root=data_root_masters, master="bias"
        )
        bias = Bias(files, config)
        bias.make_master_l1(filepath=bias_path)

    # master dark (not yet implemented)
    # for files in build_l0_file_lists('dark', mini_db=mini_db):
    #    dark_path = build_filepath(get_obs_id(files[0]), 'L1',
    #                               data_root=data_root_masters, master='dark')
    #    dark = Dark(files, config)
    #    dark.make_master_l1(filepath=dark_path)

    # master flat (not yet implemented)
    # for files in build_l0_file_lists('flat', mini_db=mini_db):
    #    flat_path = build_filepath(get_obs_id(files[0]), 'L1',
    #                               data_root=data_root_masters, master='flat')
    #    flat = Flat(files, config)
    #    flat.make_master_l1(filepath=flat_path)

    # Stack the ThAr exposures into a master wavelength solution, since the
    # emission-line spectrum anchors the per-order wavelength calibration.
    for files in build_l0_file_lists("thar", mini_db=mini_db):
        wls_master_path = build_filepath(
            get_obs_id(files[0]), "L2", data_root=data_root_masters, master="thar"
        )
        wls_diagnostics_path = (
            wls_master_path.removesuffix("_L2.fits") + "_diagnostics.h5"
        )

        wls = WLS(files, config)
        wls.make_master_l2(
            master_path=wls_master_path, diagnostics_path=wls_diagnostics_path
        )

    print("\n\n=== exiting kpf_drp_masters pipeline ===\n\n")
