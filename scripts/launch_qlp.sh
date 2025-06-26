#!/bin/bash

read -r -d '' SCRIPT_DOC <<'EOF'
Script name: launch_qlp.sh
Author: Andrew Howard

This script launches QLP (Quicklook Pipeline) instances for data levels 
L0, 2D, L1, L2, and masters. It utilizes the specified recipe and config 
files to process observational data in the KPF pipeline. The script can 
optionally process only recent observations from the current day.  
The number of cores available for each data level is 50 by default, but 
can be adjusted.

Command-line options (all are optional):
  --only_recent       Use a specialized recipe to process only observations
                      from the current day.
  --not_recent        Use a specialized recipe to process only observations
                      from more than one day ago and applies lower priority.
  --ncpu <number>     Set the number of CPUs to use (default: 50).
  -h, --help          Display this help message and exit.

Examples:
1. Launch QLP instances for all data levels with the default recipe:
   ./launch_qlp.sh

2. Launch QLP instances for only recent observations:
   ./launch_qlp.sh --only_recent

3. Launch QLP instances for observations from more than a day ago:
   ./launch_qlp.sh --not_recent

4. Launch QLP instances with 20 CPUs:
   ./launch_qlp.sh --ncpu 20

5. Display the help message:
   ./launch_qlp.sh -h
EOF

# Default configuration
recipe_file="recipes/quicklook_watch.recipe"
config_file="configs/quicklook_watch.cfg"
data_levels=("L0" "2D" "L1" "L2" "masters")
ncpus=50
nice_prefix=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --only_recent)
            recipe_file="recipes/quicklook_watch_only_recent.recipe"
            ;;
        --not_recent)
            recipe_file="recipes/quicklook_watch_not_recent.recipe"
            nice_prefix="nice -n 10"
            ;;
        --ncpu)
            shift
            ncpus=$1
            ;;
        -h|--help)
            echo "$SCRIPT_DOC"
            exit 0
            ;;
    esac
    shift
done

# Launch QLP instances for each data level
for lvl in "${data_levels[@]}"; do
    cmd="${nice_prefix} kpf --ncpus=${ncpus} --watch ${KPFPIPE_DATA}/${lvl}/ -r ${recipe_file} -c ${config_file}"
    echo $cmd
    eval $cmd &
done
