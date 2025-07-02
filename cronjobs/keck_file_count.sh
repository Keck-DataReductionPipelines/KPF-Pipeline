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
  ["L0_QLP"]="/kpfdata/data_drp/QLP/${DATE}/*/L0"
  ["2D_QLP"]="/kpfdata/data_drp/QLP/${DATE}/*/2D"
  ["L1_QLP"]="/kpfdata/data_drp/QLP/${DATE}/*/L1"
  ["L2_QLP"]="/kpfdata/data_drp/QLP/${DATE}/*/L2"
)

# Output order
ORDER=("koadata" "L0" "L0_QLP" "L1" "L1_QLP" "L2" "L2_QLP" "QLP" "2D_QLP")

log "File count for UT files: ${DATE}"
log "----------------------------------"

for key in "${ORDER[@]}"; do
  dir="${DIRS[$key]}"
  if [ "$key" == "koadata" ]; then
    # Count only *.fits files for koadata
    count=$(ls -1 "${dir}"*fits 2>/dev/null | wc -l)
  elif [[ "$key" == *_QLP ]]; then
    count=$(find $dir -type d 2>/dev/null | wc -l)
  else
    count=$(ls -1 "$dir" 2>/dev/null | wc -l)
  fi
  log "$(printf "%-10s: %3d files" "$key" "$count")"
done

log "----------------------------------"
log "Log Times"
log "----------------------------------"

log "$(ls -lt --time-style="+%Y-%m-%d %H:%M:%S" /data/data_drp/logs/${DATE}/KP*log 2>/dev/null | head -n 1 |  awk '{print $6 " " $7}') Last KP log file: $(ls -t /data/data_drp/logs/${DATE}/KP*log 2>/dev/null | head -n 1)"
log "$(tail /data/data_drp/logs/${DATE}/keck_kpf_nightly_${DATE}.log | tail -n 5)"
log "$(ls -lt --time-style="+%Y-%m-%d %H:%M:%S" /data/data_drp/logs/${DATE}/*stdout 2>/dev/null | head -n 1 |  awk '{print $6 " " $7}') Pipeline log updated"

echo "Logged count to: $OUTPUT"
