import os

from kpfpipe.modules.masters.bias import Bias
#from kpfpipe.modules.masters.dark import Dark
#from kpfpipe.modules.masters.flat import Flat
#from kpfpipe.modules.masters.wls import WLS

from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.kpf import get_obs_id
from kpfpipe.utils.pipeline import build_filepath, build_mini_database, get_calibration_stack_clusters


def main(config, args):
    print("\n\n=== entering kpf_drp_masters pipeline ===\n\n")

    datecode = args.datecode

    data_dirs = config.get_params(['DATA_DIRS'])
    data_root_in  = data_dirs['KPF_DATA_INPUT']
    data_root_out = data_dirs['KPF_DATA_OUTPUT']

    l0_filelist = os.path.join(data_root_in, 'L0', datecode)
    mini_db = build_mini_database(l0_filelist)

    # master bias
    for files in get_calibration_stack_clusters(mini_db, 'bias'):
        bias_handler = Bias(files, config)
        bias_l1 = bias_handler.make_master_l1()
        bias_l1.to_fits(build_filepath(get_obs_id(files[0]), data_root_out, 'L1', master='bias'))

    # master dark (not yet implemented)
    #for files in get_calibration_stack_clusters(mini_db, 'dark'):
    #    dark_handler = Dark(files, config)
    #    dark_l1 = dark_handler.make_master_l1()
    #    dark_l1.to_fits(build_filepath(get_obs_id(files[0]), data_root_out, 'L1', master='dark'))

    # master flat (not yet implemented)
    #for files in get_calibration_stack_clusters(mini_db, 'flat'):
    #    flat_handler = Flat(files, config)
    #    flat_l1 = flat_handler.make_master_l1()
    #    flat_l1.to_fits(build_filepath(get_obs_id(files[0]), data_root_out, 'L1', master='flat'))

    # wavelength solution (not yet implemented)
    #for files in get_calibration_stack_clusters(mini_db, 'thar-wls'):
    #    wls_handler = WLS(files, config)
    #    wls_l1 = wls_handler.make_master_l1()
    #    wls_l1.to_fits(build_filepath(get_obs_id(files[0]), data_root_out, 'L1', master='thar-wls'))

    print("\n\n=== exiting kpf_drp_masters pipeline ===\n\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='KPF masters DRP')
    parser.add_argument('-c', '--config', required=True, help='path to TOML config file')
    parser.add_argument('--datecode', required=True, help='datecode, e.g. 20240405')
    args = parser.parse_args()
    config = ConfigHandler(args.config)
    main(config, args)
