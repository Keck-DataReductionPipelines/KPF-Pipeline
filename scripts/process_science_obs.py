# this script will accept a date range OR list of obs_ids
# runs kpf_drp_science.py on all relevant files

import argparse

do argument parsing from config

if argument data_dir_input not supplied:
    data_dir_input = config.KFP_DATA_INPUT

if argument data_dir_output no supplied:
    data_dir_output = config.KPF_DATA_OUTPUT