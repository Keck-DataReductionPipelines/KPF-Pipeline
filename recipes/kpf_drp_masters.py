import os

from kpfpipe.modules.masters.bias import Bias
#from kpfpipe.modules.masters.dark import Dark
#from kpfpipe.modules.masters.flat import Flat
from kpfpipe.modules.masters.wls import WLS

from kpfpipe.utils.kpf import get_obs_id
from kpfpipe.utils.pipeline import build_filepath, build_l0_file_lists, build_mini_database


def main(config, args):
    print("\n\n=== entering kpf_drp_masters pipeline ===\n\n")

    if not args.datecode:
        raise SystemExit("Error: --datecode is required for the masters recipe (e.g. -d 20240405)")

    datecode = args.datecode

    data_dirs = config.get_params(['DATA_DIRS'])
    data_root_in  = data_dirs['KPF_DATA_INPUT']
    data_root_out = data_dirs['KPF_DATA_OUTPUT']

    l0_dir = os.path.join(data_root_in, 'L0', datecode)
    if not os.path.isdir(l0_dir):
        raise SystemExit(f"L0 data directory not found: {l0_dir}")
    mini_db = build_mini_database(l0_dir)

    # master bias
    for files in build_l0_file_lists('bias', mini_db=mini_db):
        bias_path = build_filepath(get_obs_id(files[0]), 'L1', data_root=data_root_out, master='bias')
        bias_handler = Bias(files, config)
        bias_handler.make_master_l1(filepath=bias_path)

    # master dark (not yet implemented)
    #for files in build_l0_file_lists('dark', mini_db=mini_db):
    #    dark_path = build_filepath(get_obs_id(files[0]), 'L1', data_root=data_root_out, master='dark')
    #    dark_handler = Dark(files, config)
    #    dark_handler.make_master_l1(filepath=dark_path)

    # master flat (not yet implemented)
    #for files in build_l0_file_lists('flat', mini_db=mini_db):
    #    flat_path = build_filepath(get_obs_id(files[0]), 'L1', data_root=data_root_out, master='flat')
    #    flat_handler = Flat(files, config)
    #    flat_handler.make_master_l1(filepath=flat_path)

    # master wavelength solution (ThAr)
    for files in build_l0_file_lists('thar', mini_db=mini_db):
        wls_master_path      = build_filepath(get_obs_id(files[0]), 'L2', data_root=data_root_out, master='thar')
        wls_diagnostics_path = wls_master_path[:-len('_L2.fits')] + '_diagnostics.h5'

        wls_handler = WLS(files, config)
        wls_handler.make_master_l2(master_path=wls_master_path, diagnostics_path=wls_diagnostics_path)

    print("\n\n=== exiting kpf_drp_masters pipeline ===\n\n")

