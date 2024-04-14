#!/bin/bash

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "This script launches 15 QLP instances for L0/2D/L1/L2/masters"
    echo
    echo "Options:"
    echo "  --only_recent  Only process QLP for observations in the day."
    echo "  -h, --help     Display this help message and exit."
    echo
}

recipe_file="recipes/quicklook_watch.recipe"
config_file="configs/quicklook_watch.cfg"
data_levels=("L0" "2D" "L1" "L2" "masters")

for arg in "$@"; do
    case "$arg" in
        --only_recent)
            recipe_file="recipes/quicklook_watch_only_recent.recipe"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
    esac
done

for lvl in "${data_levels[@]}"; do
    ncpus=15
    cmd="kpf --ncpus=${ncpus} --watch ${KPFPIPE_DATA}/${lvl}/ -r ${recipe_file} -c {config_file}"
    eval $cmd &
done
