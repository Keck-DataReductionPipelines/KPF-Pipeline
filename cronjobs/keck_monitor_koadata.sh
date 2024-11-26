#!/bin/bash

# Copy the NFS mounted data to a local data directory
BASE_WATCH_DIR="/koadata/KPF"
BASE_DEST_DIR="/kpfdata/data_workspace/L0"

# Default HISTORY time
TIME_HISTORY=5

# Check for the --fullnight option in the command line arguments
if [[ "$1" == "--fullnight" ]]; then
    # will not only look for new files
    echo "Full night option selected, checking the entire directory not only new files."
    TIME_HISTORY=0  
else
    echo "Monitoring recent files (last $TIME_HISTORY minutes) by default."
fi

# get the current UTC date-based directories
get_watch_dir() {
    CURRENT_DATE=$(date -u +"%Y%m%d")
    echo "${BASE_WATCH_DIR}/${CURRENT_DATE}/lev0"
}

get_dest_dir() {
    CURRENT_DATE=$(date -u +"%Y%m%d")
    echo "${BASE_DEST_DIR}/${CURRENT_DATE}"
}

get_parent_dest_dir() {
    CURRENT_DATE=$(date -u +"%Y%m%d")
    echo "${BASE_DEST_DIR}/${CURRENT_DATE}"
}

# Initial directories
WATCH_DIR=$(get_watch_dir)
DEST_DIR=$(get_dest_dir)
PARENT_DEST_DIR=$(get_parent_dest_dir)

# Ensure destination directories exist
mkdir -p "$DEST_DIR"

# Ensure the initial watch directory exists
if [ ! -d "$WATCH_DIR" ]; then
    echo "Directory $WATCH_DIR does not exist!"
    exit 1
fi

echo "Monitoring directory: $WATCH_DIR (UTC)"
echo "FITS files copied to: $DEST_DIR (UTC)"

# Start monitoring with a loop that checks for new files
while true
do
    # Check if the date has changed
    NEW_WATCH_DIR=$(get_watch_dir)
    NEW_DEST_DIR=$(get_dest_dir)

    if [ "$NEW_WATCH_DIR" != "$WATCH_DIR" ]; then
        echo "UTC Date changed. Now monitoring directory: $NEW_WATCH_DIR"

        NEW_PARENT_DEST_DIR=$(get_parent_dest_dir)
        WATCH_DIR="$NEW_WATCH_DIR"
        DEST_DIR="$NEW_DEST_DIR"
        PARENT_DEST_DIR="$NEW_PARENT_DEST_DIR"

        # Ensure the new parent destination directory exists
        mkdir -p "$PARENT_DEST_DIR"

        # Ensure the new destination directory exists
        mkdir -p "$DEST_DIR"

        # Ensure the new watch directory exists before proceeding
        if [ ! -d "$WATCH_DIR" ]; then
            echo "Directory $WATCH_DIR does not exist, waiting for it to be created..."
            continue
        fi
    fi

    # Poll for new .fits files
        if [ "$TIME_HISTORY" -eq 0 ]; then
        # Full night option: search the entire directory
        find "$WATCH_DIR" -type f -name "*.fits*" | while read -r NEW_FILE; do
            # Get the base filename to check existence in the destination directory
            FILE_NAME=$(basename "$NEW_FILE")
            DEST_FILE="$DEST_DIR/$FILE_NAME"

            # Check if the file already exists in the destination
            if [ ! -e "$DEST_FILE" ]; then
                # Copy the new .fits file to the destination directory
                cp "$NEW_FILE" "$DEST_DIR"
                echo "New .fits file '$NEW_FILE' copied to $DEST_DIR at $(date -u)"
            else
                echo "File '$DEST_FILE' already exists, skipping."
            fi
        done
    else
        # Default option: search for files modified in the last $TIME_HISTORY minutes
        find "$WATCH_DIR" -type f -name "*.fits*" -mmin -"$TIME_HISTORY" | while read -r NEW_FILE; do
            # Get the base filename to check existence in the destination directory
            FILE_NAME=$(basename "$NEW_FILE")
            DEST_FILE="$DEST_DIR/$FILE_NAME"

            # Check if the file already exists in the destination
            if [ ! -e "$DEST_FILE" ]; then
                # Copy the new .fits file to the destination directory
                cp "$NEW_FILE" "$DEST_DIR"
                echo "New .fits file '$NEW_FILE' copied to $DEST_DIR at $(date -u)"
            fi
        done
    fi
    # wait before checking again
    sleep 60
done
