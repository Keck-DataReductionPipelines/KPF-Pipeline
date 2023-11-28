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
#   or files of those types with older modification dates, the script checks 
#   the 'GREEN' and 'RED' keywords in the FITS header of the L0 file 
#   and excludes files from the missing count if the Green and Red cameras 
#   are both not selected. The script outputs a summary for each YYYYMMDD 
#   directory, showing the count of such files and the most recent 
#   L0 modification date. The script takes a starting date (YYYYMMDD) as an 
#   argument and optionally an end date and flags to print missing files and
#   touch the base L0 files of missing 2D/L1/L2 files.
#
# Options:
#   --help           Display this message
#   --print_missing  Display missing file names
#   --touch_missing  Touch the base L0 files of missing 2D/L1/L2 files
#
# Usage:
#   ./kpf_processing_progress.sh YYYYMMDD [YYYYMMDD] [--print_missing]
#
# Example:
#   ./kpf_processing_progress.sh 20231114 20231231 --print_missing
# === HELP END ===

# To-do:
#   * - add a check for software version (require the latest version of master?)
#   * - have the script ignore files marked as junk

# Check for --help argument
if [[ "$1" == "--help" ]]; then
    awk '/^# === HELP START ===/,/^# === HELP END ===/' "$0" | sed -e '1d;$d' -e 's/^#//; s/^ //' 
    exit 0
fi

# Initialize flags
print_missing=false
touch_missing=false

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
            shift
            ;;
        --touch_missing)
            touch_missing=true
            shift
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

# Array to store base files for touching
declare -a missing_base_files

# Function to check GREEN, RED, CA_HK keywords - returns 0 (success) if any camera was used
# the logic is a bit awkward, but is optimized for (relative) speed; fitsheader is run a minimum number of times
function green_red_cahk_present {
    local file_path=$1
    
    green_keyword=$(fitsheader --extension 0 -k GREEN "$file_path" | grep YES | wc -l)
    if [[ $green_keyword == "1" ]] ; then
        return 0 
    fi
    
    red_keyword=$(fitsheader --extension 0 -k RED "$file_path" | grep YES | wc -l)
    if [[ $red_keyword == "1" ]] ; then
        return 0
    fi
    
    cahk_keyword=$(fitsheader --extension 0 -k CA_HK "$file_path" | grep YES | wc -l)
    if [[ $cahk_keyword == "1" ]] ; then
        return 0
    else
        return 1
    fi
}

function not_bias_or_dark {
    local file_path=$1

    # Get the IMTYPE keyword value
    imtype_keyword=$(fitsheader --extension 0 -k IMTYPE "$file_path" | awk '{print $3}')

    if [[ $imtype_keyword == *"Dark"* ]]; then
        return 1 # dark detected
    elif [[ $imtype_keyword == *"Bias"* ]]; then
        return 1 # bias detected
    else
        return 0 # success - not a dark or bias
    fi
}


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
        total_count_2D=0
        total_count_L1=0
        total_count_L2=0
        match_count_2D=0
        match_count_L1=0
        match_count_L2=0
        recent_mod_date=0

        # Loop through each .fits file in the L0/YYYYMMDD directory
        for file in "$dir/KP.$date_code."?????.??.fits; do
            if [ -f "$file" ]; then
                # Increment total file count
                ((total_count++))
                ((total_count_2D++))
                ((total_count_L1++))
                ((total_count_L2++))

                # Get the modification date of the L0 file
                mod_date_L0=$(date -r "$file" "+%s")
                if [ "$mod_date_L0" -gt "$recent_mod_date" ]; then
                    recent_mod_date="$mod_date_L0"
                fi

                # Construct the corresponding 2D, L1, and L2 filenames
                file_2d="$base_dir/2D/$date_code/$(basename "${file%.fits}")_2D.fits"
                file_L1="$base_dir/L1/$date_code/$(basename "${file%.fits}")_L1.fits"
                file_L2="$base_dir/L2/$date_code/$(basename "${file%.fits}")_L2.fits"

                # 2D file logic
                if [ ! -f "$file_2d" ]; then
                    if $print_missing; then
                        if green_red_cahk_present "$file"; then
                            echo "Missing 2D file: $file_2d"
                        fi
                    fi
                    if $touch_missing; then
                        if green_red_cahk_present "$file"; then
                            missing_base_files+=("$file")
                        fi
                    fi 
                elif [ $(date -r "$file_2d" "+%s") -lt "$mod_date_L0" ]; then
                    if $print_missing; then
                        if green_red_cahk_present "$file"; then
                            echo "Old 2D file: $file_2d"
                        fi
                    fi
                    if $touch_missing; then
                        if green_red_cahk_present "$file"; then
                            missing_base_files+=("$file")
                        fi
                    fi 
                elif [ $(date -r "$file_2d" "+%s") -gt "$mod_date_L0" ]; then
                    ((match_count_2D++))
                fi

                # L1 file logic
                if [ ! -f "$file_L1" ]; then
                    if $print_missing; then
                        if green_red_cahk_present "$file"; then
                            echo "Missing L1 file: $file_L1"
                        fi
                    fi
                    if $touch_missing; then
                        if green_red_cahk_present "$file"; then
                            missing_base_files+=("$file")
                        fi
                    fi 
                    if ! green_red_cahk_present "$file"; then
                        ((total_count_L1--))
                    fi 
                elif [ $(date -r "$file_L1" "+%s") -lt "$mod_date_L0" ]; then
                    if $print_missing; then
                        if green_red_cahk_present "$file"; then
                            echo "Old L1 file: $file_L1"
                        fi
                    fi
                    if $touch_missing; then
                        if green_red_cahk_present "$file"; then
                            missing_base_files+=("$file")
                        fi
                    fi 
                elif [ $(date -r "$file_L1" "+%s") -gt "$mod_date_L0" ]; then
                    ((match_count_L1++))
                fi

                # L2 file logic
                if [ ! -f "$file_L2" ]; then
                    if $print_missing; then
                        if green_red_cahk_present "$file"; then
                            if not_bias_or_dark "$file"; then
                                echo "Missing L2 file: $file_L2"
                            fi
                        fi
                    fi
                    if $touch_missing; then
                        if green_red_cahk_present "$file"; then
                            if not_bias_or_dark "$file"; then
                                missing_base_files+=("$file")
                            fi
                        fi
                    fi
                    if ! green_red_cahk_present "$file" || ! not_bias_or_dark "$file"; then
                        ((total_count_L2--))
                    fi 
                elif [ $(date -r "$file_L2" "+%s") -lt "$mod_date_L0" ]; then
                    if $print_missing; then
                        if green_red_cahk_present "$file"; then
                            if not_bias_or_dark "$file"; then
                                echo "Old L2 file: $file_L2"
                            fi
                        fi
                    fi
                    if $touch_missing; then
                        if green_red_cahk_present "$file"; then
                            if not_bias_or_dark "$file"; then
                                missing_base_files+=("$file")
                            fi
                        fi
                    fi 
                elif [ $(date -r "$file_L2" "+%s") -gt "$mod_date_L0" ]; then
                    ((match_count_L2++))
                fi
            fi
        done

        # Format the most recent modification date without seconds
        formatted_recent_mod_date=$(date -d "@$recent_mod_date" "+%Y-%m-%d %H:%M")

        # Calculate percentage and print summary
        if [ $total_count -gt 0 ]; then
            percentage_2D=$((match_count_2D * 100 / total_count_2D))
            percentage_L1=$((match_count_L1 * 100 / total_count_L1))
            percentage_L2=$((match_count_L2 * 100 / total_count_L2))
            printf "%-8s | %-16s | %4d/%-4d %3d%% | %4d/%-4d %3d%% | %4d/%-4d %3d%%\n" "$date_code" "$formatted_recent_mod_date" "$match_count_2D" "$total_count_2D" "$percentage_2D" "$match_count_L1" "$total_count_L1" "$percentage_L1" "$match_count_L2" "$total_count_L2" "$percentage_L2"
        fi
    fi
done
printf "%s\n" "------------------------------------------------------------------------------"

# Touch missing files if option set
if $touch_missing && [ ${#missing_base_files[@]} -gt 0 ]; then
    uniq_missing_base_files=($(for file in "${missing_base_files[@]}"; do echo "${file}"; done | sort -u))
    echo "The following L0 files have missing corresponding 2D, L1, or L2 files:"
    printf "%s\n" "${uniq_missing_base_files[@]}"
    read -p "Do you want to touch these files? [y/N] " confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
        for file in "${uniq_missing_base_files[@]}"; do
            echo "touch $file"
            touch "$file"
            sleep 0.2
        done
    fi
elif $touch_missing; then
    echo
    echo "All files are up to date."
fi

echo $missing_base_files

