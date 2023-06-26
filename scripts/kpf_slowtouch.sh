#!/bin/bash

read -r -d '' SCRIPT_DOC <<'EOF'
Script name: kpf_slowtouch.sh
Author: Andrew Howard
        with assistance from Chat-GPT4
        or maybe the other way around
Date: June 23, 2023

This script is used to touch a list of KPF L0 files that have names like 
KP.20230623.12345.67.fits.  This is useful to initiate reprocessing 
using the KPF DRP.  The list of L0 files can be provided in multiple ways:
   1. As command-line arguments when invoking the script.
   2. In the first column of a CSV file specified with the -f option.
      This is useful for CSV files with a large set of L0 filenames 
      downloaded from Jump.  Such files might have double quotes around 
      the L0 filename, which the script will remove when appropriate.
   3. All filenames in a directory specified with the -d option.

Command-line options (all are optional):
-f <filename>       : The script will read the KPF L0 filenames 
                      from the first column of a CSV with the name <filename>.
                      Useful for lists of L0 files downloaded from Jump.
-d <directory>      : Adds every file in <directory> to the list of L0 files.
-p <path>           : Sets the L0 path to <path>.
                      Default value: /data/kpf/L0
-s <sleep_interval> : Sets the interval between file touches.
                      Default value: 0.2 [sec]
-e                  : Echo the touch commands instead of executing them.

Examples:
1. To provide filenames using command line arguments:
   ./kpf_slowtouch.sh KP.20230623.12345.67.fits KP.20230623.12345.68.fits
2. To provide filenames using a CSV file:
   ./kpf_slowtouch.sh -f filenames.csv
3. To provide files listed in a directory:
   ./kpf_slowtouch.sh -d /path/to/directory
4. To change the default L0 path and sleep interval between touches:
   ./kpf_slowtouch.sh KP.20230623.12345.67.fits -p /new/path -s 0.5
5. To echo the touch commands instead of executing them:
   ./kpf_slowtouch.sh KP.20230623.12345.67.fits -e
EOF

# Default values
l0_path="/data/kpf/L0"
sleep_interval=0.2

if [[ $# -eq 0 ]]; then
    echo "No arguments supplied. Please provide -f filename.csv, -d directory, or KPF filenames as command line arguments."
    exit 1
fi

# Declare an empty array to store the KPF filenames
declare -a filenames=()

# Flags to keep track of command line options
csv_processed=0
l0_path_processed=0
sleep_interval_processed=0
directory_processed=0
echo_mode=0

# Process all the arguments
for arg in "$@"
do
    # Check for help option
    if [[ $arg == "-h" ]] || [[ $arg == "--help" ]]; then
        echo "$SCRIPT_DOC"
        exit 0
    fi

    # Check for echo option
    if [[ $arg == "-e" ]]; then
        echo_mode=1
        continue
    fi

    # Check if the "-f" option was used
    if [[ $arg == "-f" ]]; then
        csv_processed=1
        continue
    fi

    # Check if the "-p" option was used
    if [[ $arg == "-p" ]]; then
        l0_path_processed=1
        continue
    fi

    # Check if the "-s" option was used
    if [[ $arg == "-s" ]]; then
        sleep_interval_processed=1
        continue
    fi

    # Check if the "-d" option was used
    if [[ $arg == "-d" ]]; then
        directory_processed=1
        continue
    fi

    # Check if the previous argument was "-f", indicating that this argument is CSV filename
    if [[ $csv_processed -eq 1 ]]; then
        filename=$arg
        if [[ ! -f $filename ]]; then
            echo "File $filename does not exist."
            exit 1
        fi
        # Use awk to get the first column of the csv, removing quotes and 'observation_id' strings
        # Add the filenames to the array
        while IFS= read -r line
        do
            filenames+=("$line")
        done < <(awk -F',' '{ gsub(/"/,""); if ($1 !~ /^observation_id/) print $1 }' "$filename")
        csv_processed=0
        continue
    fi

    # Check if the previous argument was "-p", indicating that this argument is the l0_path
    if [[ $l0_path_processed -eq 1 ]]; then
        l0_path=$arg
        l0_path_processed=0
        continue
    fi

    # Check if the previous argument was "-s", indicating that this argument is the sleep_interval
    if [[ $sleep_interval_processed -eq 1 ]]; then
        sleep_interval=$arg
        sleep_interval_processed=0
        continue
    fi

    # Check if the previous argument was "-d", indicating that this argument is a directory
    if [[ $directory_processed -eq 1 ]]; then
        directory=$arg
        if [[ ! -d $directory ]]; then
            echo "Directory $directory does not exist."
            exit 1
        fi
        # Use ls to get the filenames in the directory and add them to the array
        while IFS= read -r file
        do
            filenames+=("$file")
        done < <(ls "$directory")
        directory_processed=0
        continue
    fi

    # The argument is a filename
    text=${arg//\"/}  # Remove quotes (which are present in KPF L0 lists downloaded from Jump)
    if [[ $text != observation_id* ]]; then
        filenames+=("$text")
    fi
done

# Check if "-f" or "-d" was used but no following argument was provided
if [[ $csv_processed -eq 1 ]]; then
    echo "No filename supplied. Please provide a filename after -f."
    exit 1
fi

# Iterate over the array of KPF filenames and touch/echo them
for fn in "${filenames[@]}"
do
    fn=${fn%.fits}  # Remove ".fits" from filename (needed to make the -d and -f options compatible)
    yyyymmdd=${fn:3:8}
    fullpath=$l0_path/$yyyymmdd/$fn.fits
    if [[ $echo_mode -eq 1 ]]; then
        echo "touch $fullpath"
    else
        echo "$(date +%T) touching file: $fullpath"
        touch "$fullpath"
        sleep $sleep_interval
    fi
done
