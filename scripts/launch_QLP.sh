#!/bin/bash

# This script is designed to launch the 5 QLP instances with a single command

ncpus=1
data_levels=("L0" "2D" "L1" "L2" "masters")

for lvl in "${data_levels[@]}"; do
    cmd="kpf --ncpus=${ncpus} --watch ${KPFPIPE_DATA}/${lvl}/ -r recipes/quicklook_watch_dir.recipe -c configs/quicklook_watch_dir.cfg"
    $cmd &
done
