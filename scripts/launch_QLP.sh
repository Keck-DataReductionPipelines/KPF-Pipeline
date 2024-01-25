#!/bin/bash

# This script is designed to launch the 5 QLP instances with a single command

data_levels=("L0" "2D" "L1" "L2" "masters")

for lvl in "${data_levels[@]}"; do
    if lvl=='L1':
        ncpus=20
    else:
        ncpus=10
    cmd="kpf --ncpus=${ncpus} --watch ${KPFPIPE_DATA}/${lvl}/ -r recipes/quicklook_watch.recipe -c configs/quicklook_watch.cfg"
    $cmd &
done
