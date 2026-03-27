import os

from kpfpipe.modules.masters.bias import Bias
#from kpfpipe.modules.masters.dark import Dark
#from kpfpipe.modules.masters.flat import Flat
#from kpfpipe.modules.masters.wls import WLS

from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.pipeline import build_filepath, build_mini_database, build_l0_file_list


def main(config, args):
    print("\n\n=== entering kpf_drp_masters pipeline ===\n\n")

    datecode = args.datecode

    data_dirs = config.get_params(['DATA_DIRS'])
    data_root_in  = data_dirs['KPF_DATA_INPUT']
    data_root_out = data_dirs['KPF_DATA_OUTPUT']
    data_dir_l0 = os.path.join(data_root_in, 'L0', datecode)

    # scan headers and write mini database CSV to data_dir
    build_mini_database(data_dir_l0)

    # master bias
    bias_handler = Bias(build_l0_file_list(data_dir_l0, 'bias'), config)
    bias_l1 = bias_handler.make_master_l1()
    bias_l1.to_fits(build_filepath(datecode, data_root_out, 'L1', master='bias'))

    # master dark (not yet implemented)
    #dark_handler = Dark(build_l0_file_list(data_dir_l0, 'dark'), config)
    #dark_l1 = dark_handler.make_master_l1()
    #dark_l1.to_fits(build_filepath(datecode, data_root_out, 'L1', master='dark'))

    # master flat (not yet implemented)
    #flat_handler = Flat(build_l0_file_list(data_dir_l0, 'flat'), config)
    #flat_l1 = flat_handler.make_master_l1()
    #flat_l1.to_fits(build_filepath(datecode, data_root_out, 'L1', master='flat'))

    # wavelength solution (not yet implemented)
    #wls = WLS(build_l0_file_list(data_dir_l0, 'thar-wls'), config)
    #wls_l1 = wls.make_master_l1()
    #wls_l1.to_fits(build_filepath(datecode, data_root_out, 'L1', master='thar-wls'))

    print("\n\n=== exiting kpf_drp_masters pipeline ===\n\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='KPF masters DRP')
    parser.add_argument('-c', '--config', required=True, help='path to TOML config file')
    parser.add_argument('--datecode', required=True, help='datecode, e.g. 20240405')
    args = parser.parse_args()
    config = ConfigHandler(args.config)
    main(config, args)
