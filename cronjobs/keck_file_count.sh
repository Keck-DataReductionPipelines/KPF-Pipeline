#!/bin/bash

HR_MIN=$(date +"%Hh%Mm")
DATE=$(date +"%Y%m%d")

OUTPUT="/kpfdata/logs/DailyRuns/file_count_${DATE}_${HR_MIN}.log"

# output to log and standard out
log() {
    echo "$@" | tee -a "$OUTPUT"
}

log "----------------------------------"
log "Counted at: ${DATE}_${HR_MIN}"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --date)
            DATE="$2"
            shift
            ;;
        --yesterday)
            DATE=$(date --date="yesterday" +"%Y%m%d")
            ;;
        --nolog)
            OUTPUT="/dev/stdout"
            ;;
    esac
    shift
done

# Directories to count
declare -A DIRS=(
  ["koadata"]="/koadata/KPF/${DATE}/lev0/"
  ["L0"]="/kpfdata/data_workspace/L0/${DATE}/"
  ["L1"]="/kpfdata/data_drp/L1/${DATE}/"
  ["L2"]="/kpfdata/data_drp/L2/${DATE}/"
  ["QLP"]="/kpfdata/data_drp/QLP/${DATE}/"
)

# Output order
ORDER=("koadata" "L0" "L1" "L2" "QLP")

log "File count for UT files: ${DATE}"
log "----------------------------------"

for key in "${ORDER[@]}"; do
  dir="${DIRS[$key]}"
  if [ "$key" == "koadata" ]; then
    # Count only *.fits files for koadata
    count=$(ls -1 "${dir}"*fits 2>/dev/null | wc -l)
  else
    count=$(ls -1 "$dir" 2>/dev/null | wc -l)
  fi
  log "$(printf "%-10s: %3d files" "$key" "$count")"
done

log "Logged count to: $OUTPUT"
