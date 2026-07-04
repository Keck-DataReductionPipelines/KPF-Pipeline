#!/usr/bin/env bash
#
# fetch_l0.sh — retrieve KPF L0 files for a list of obs_ids from a remote server.
#
# Reads a text file of obs_ids (one per line, optionally preceded by an index
# column, e.g. "1<TAB>KP.20240727.56085.97") and rsyncs the matching L0 FITS
# files from a remote L0 root into a local L0 root, preserving the
# {ROOT}/{datecode}/{obs_id}.fits layout on both ends. The datecode is the
# middle field of the obs_id (KP.<datecode>.<seconds>...).
#
# The remote and local L0 roots are supplied as arguments (or via the
# KPF_REMOTE_L0 / KPF_LOCAL_L0 environment variables) — nothing is tied to a
# particular machine or account.
#
# Usage:
#   fetch_l0.sh -r <remote_L0_root> -l <local_L0_root> <obs_id_list.txt>
#
# Examples:
#   fetch_l0.sh -r user@host:/data/kpf/L0 -l "$HOME/data/kpf/L0" obs_ids.txt
#   fetch_l0.sh -r shrek:/data/kpf/L0     -l "$HOME/data/kpf/L0" obs_ids.txt  # ssh alias
#   KPF_REMOTE_L0=shrek:/data/kpf/L0 KPF_LOCAL_L0="$HOME/data/kpf/L0" \
#       fetch_l0.sh obs_ids.txt
#
set -euo pipefail

REMOTE_ROOT="${KPF_REMOTE_L0:-}"
LOCAL_ROOT="${KPF_LOCAL_L0:-}"

usage() {
    cat >&2 <<EOF
Usage: $0 -r <remote_L0_root> -l <local_L0_root> <obs_id_list.txt>

  -r  remote L0 root as an rsync/ssh spec, e.g. user@host:/data/kpf/L0
      (or an ssh-config alias like shrek:/data/kpf/L0).
      Falls back to \$KPF_REMOTE_L0.
  -l  local L0 root, e.g. \$HOME/data/kpf/L0.  Falls back to \$KPF_LOCAL_L0.

Files are transferred as {root}/{datecode}/{obs_id}.fits on both ends.
EOF
    exit 2
}

while getopts ":r:l:h" opt; do
    case "$opt" in
        r) REMOTE_ROOT="$OPTARG" ;;
        l) LOCAL_ROOT="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

[[ $# -eq 1 ]] || usage
LIST_FILE="$1"
[[ -n "$REMOTE_ROOT" ]] || { echo "Error: remote L0 root not set (use -r or \$KPF_REMOTE_L0)" >&2; usage; }
[[ -n "$LOCAL_ROOT" ]]  || { echo "Error: local L0 root not set (use -l or \$KPF_LOCAL_L0)" >&2; usage; }
[[ -f "$LIST_FILE" ]]   || { echo "Error: no such file: $LIST_FILE" >&2; exit 1; }

# Strip any trailing slash so the {root}/{datecode} joins stay clean.
REMOTE_ROOT="${REMOTE_ROOT%/}"
LOCAL_ROOT="${LOCAL_ROOT%/}"

# Extract obs_ids: pull the KP.<date>.<secs> token from each line, ignoring the
# index column and any blank/garbage lines. (read loop, not mapfile, so this
# works on the bash 3.2 that ships with macOS.)
OBS_IDS=()
while IFS= read -r obs_id; do
    OBS_IDS+=("$obs_id")
done < <(grep -oE 'KP\.[0-9]{8}\.[0-9]+(\.[0-9]+)?' "$LIST_FILE")

n_total=${#OBS_IDS[@]}
if [[ $n_total -eq 0 ]]; then
    echo "Error: no obs_ids found in $LIST_FILE" >&2
    exit 1
fi
echo "Found $n_total obs_ids in $LIST_FILE"
echo "  remote: $REMOTE_ROOT"
echo "  local:  $LOCAL_ROOT"

n_ok=0
n_fail=0
failed=()

for obs_id in "${OBS_IDS[@]}"; do
    datecode="${obs_id#KP.}"      # strip leading "KP."
    datecode="${datecode%%.*}"    # keep everything before the next "."

    remote_path="${REMOTE_ROOT}/${datecode}/${obs_id}.fits"
    local_dir="${LOCAL_ROOT}/${datecode}"
    mkdir -p "$local_dir"

    echo "[$((n_ok + n_fail + 1))/$n_total] $obs_id"
    # -a preserve attrs, -z compress, --partial resume, --progress per-file bar.
    # rsync skips files already present and up to date, so re-runs are cheap.
    # (Avoids --info=progress2, which needs rsync 3.x; macOS ships 2.6.9.)
    if rsync -az --partial --progress \
        "${remote_path}" "${local_dir}/"; then
        n_ok=$((n_ok + 1))
    else
        echo "  WARNING: failed to fetch $obs_id" >&2
        n_fail=$((n_fail + 1))
        failed+=("$obs_id")
    fi
done

echo
echo "Done: $n_ok fetched, $n_fail failed (of $n_total)."
if [[ $n_fail -gt 0 ]]; then
    printf '  failed: %s\n' "${failed[@]}" >&2
    exit 1
fi
