#!/bin/bash

# === HELP START ===
# Script Name: kpf_processing_progress.sh
#
# Description:
#   This script searches through /data/kpf/L0/YYYYMMDD subdirectories for L0 
#   files matching the pattern KP.YYYYMMDD.NNNNN.NN.fits. It records the most 
#   recent modification date of any L0 file and checks if corresponding 
#   KP.YYYYMMDD.NNNNN.NN_2D.fits, KP.YYYYMMDD.NNNNN.NN_L1.fits, and 
#   KP.YYYYMMDD.NNNNN.NN_L2.fits files in respective directories have a file 
#   modification date after the L0 file. For any missing 2D, L1, and L2 files, 
#   the script checks the 'TRIGTARG' keyword in the FITS header of the L0 file 
#   and excludes the file from the missing count if the keyword does not 
#   contain 'Green' or 'Red'. The script outputs a summary for each YYYYMMDD 
#   directory, showing the count of such files and the most recent L0 
#   modification date. The script takes a starting date (YYYYMMDD) as an 
#   argument and optionally an end date and a flag to print missing files.
#
# Options:
#   --help           Display this message
#   --print_missing  Display missing file names
#
# Usage:
#   ./check_fits_files.sh YYYYMMDD [YYYYMMDD] [--print_missing]
#
# Example:
#   ./check_fits_files.sh 20231114 20231231 --print_missing
# === HELP END ===

# Check for --help argument
if [[ "$1" == "--help" ]]; then
    awk '/^# === HELP START ===/,/^# === HELP END ===/' "$0" | sed -e '1d;$d' -e 's/^#//; s/^ //' 
    exit 0
fi

# Initialize flag for printing missing files
print_missing=false

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 YYYYMMDD [YYYYMMDD] [--print_missing]"
    exit 1
fi

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --print_missing)
            print_missing=true
            shift # Remove --print_missing from processing
            ;;
        *)
            # Remaining arguments are assumed to be dates
            if [ -z "$start_date" ]; then
                start_date=$arg
            elif [ -z "$end_date" ]; then
                end_date=$arg
            fi
            ;;
    esac
done

end_date=${end_date:-99999999} # Default to a high date if end_date is not set

# Directory path
base_dir="/data/kpf"

# Print header
printf "%s\n" 
printf "%-8s | %-16s | %-14s | %-14s | %-14s\n" "DATECODE" "LAST L0 MOD DATE" "2D PROCESSING" "L1 PROCESSING" "L2 PROCESSING"
printf "%s\n" "------------------------------------------------------------------------------"

# Loop through each subdirectory in L0
for dir in "$base_dir/L0/"????????; do
    # Extract the date code from the directory name
    date_code=$(basename "$dir")

    # Check if directory name matches YYYYMMDD format and is within the specified date range
    if [[ $date_code =~ ^[0-9]{8}$ ]] && [[ $date_code -ge $start_date ]] && [[ $date_code -le $end_date ]]; then

        # Initialize counters and most recent mod date variable
        total_count=0
        match_count_2D=0
        match_count_L1=0
        match_count_L2=0
        recent_mod_date=0

        # Loop through each .fits file in the L0/YYYYMMDD directory
        for file in "$dir/KP.$date_code."*.fits; do
            if [ -f "$file" ]; then
                # Increment total file count
                ((total_count++))

                # Get the modification date of the L0 file
                mod_date_L0=$(date -r "$file" "+%s")
                if [ "$mod_date_L0" -gt "$recent_mod_date" ]; then
                    recent_mod_date="$mod_date_L0"
                fi

                # Construct the corresponding _2D.fits, L1, and L2 filenames
                file_2d="$base_dir/2D/$date_code/$(basename "${file%.fits}")_2D.fits"
                file_L1="$base_dir/L1/$date_code/$(basename "${file%.fits}")_L1.fits"
                file_L2="$base_dir/L2/$date_code/$(basename "${file%.fits}")_L2.fits"

                # Function to check TRIGTARG keyword
                function check_trigtarg {
                    local file_path=$1
                    local type=$2
                    trig_target=$(fitsheader -k TRIGTARG "$file" | awk '{print $3}')
                    if [[ $trig_target != *"Green"* ]] && [[ $trig_target != *"Red"* ]]; then
                        echo "Excluded missing $type file (TRIGTARG not Green/Red): $file_path"
                        return 1
                    else
                        return 0
                    fi
                }

                # Check for missing files and handle TRIGTARG keyword
                if [ ! -f "$file_2d" ]; then
                    if $print_missing && check_trigtarg "$file_2d" "2D"; then
                        echo "Missing 2D file: $file_2d"
                        ((match_count_2D++))
                    fi
                elif [ $(date -r "$file_2d" "+%s") -gt "$mod_date_L0" ]; then
                    ((match_count_2D++))
                fi

                if [ ! -f "$file_L1" ]; then
                    if $print_missing && check_trigtarg "$file_L1" "L1"; then
                        echo "Missing L1 file: $file_L1"
                        ((match_count_L1++))
                    fi
                elif [ $(date -r "$file_L1" "+%s") -gt "$mod_date_L0" ]; then
                    ((match_count_L1++))
                fi

                if [ ! -f "$file_L2" ]; then
                    if $print_missing && check_trigtarg "$file_L2" "L2"; then
                        echo "Missing L2 file: $file_L2"
                        ((match_count_L2++))
                    fi
                elif [ $(date -r "$file_L2" "+%s") -gt "$mod_date_L0" ]; then
                    ((match_count_L2++))
                fi
            fi
        done

        # Format the most recent modification date without seconds
        formatted_recent_mod_date=$(date -d "@$recent_mod_date" "+%Y-%m-%d %H:%M")

        # Calculate percentage and print summary if there are any .fits files
        if [ $total_count -gt 0 ]; then
            percentage_2D=$((match_count_2D * 100 / total_count))
            percentage_L1=$((match_count_L1 * 100 / total_count))
            percentage_L2=$((match_count_L2 * 100 / total_count))
            printf "%-8s | %-16s | %4d/%-4d %3d%% | %4d/%-4d %3d%% | %4d/%-4d %3d%%\n" "$date_code" "$formatted_recent_mod_date" "$match_count_2D" "$total_count" "$percentage_2D" "$match_count_L1" "$total_count" "$percentage_L1" "$match_count_L2" "$total_count" "$percentage_L2"
        fi
    fi
done
printf "%s\n" "------------------------------------------------------------------------------"
