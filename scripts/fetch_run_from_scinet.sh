#!/usr/bin/env bash
set -euo pipefail

# fetch_run_from_scinet.sh
# Copies files/directories FROM the remote host TO this machine.
#
# Two usage modes:
# 1) Experiment mode (simple): provide the experiment folder name, and this script
#    fetches the single checkpoint file nested at:
#      /scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/<name>/ckpts/simclr/last.pt
#    (Fallback: tries cpkts if ckpts is not found.)
#    Example:
#      ./fetch_run_from_scinet.sh my_experiment_name [optional-local-dest]
#    If local dest is omitted, the file is saved to:
#      ./ckpts/simclr/<name>/last.pt
#
# 2) Raw rsync mode (advanced): provide explicit remote and local paths, optionally
#    with exclude extensions. This preserves existing behavior.
#    Example:
#      ./fetch_run_from_scinet.sh /remote/path /local/path [--exclude-ext ext1,ext2,...]

SSH_CONFIG="/ssh-keys/config"
SSH_KEY="/ssh-keys/id_ed25519"
REMOTE_USER_HOST="kostouso@trillium-gpu.scinet.utoronto.ca"

REMOTE_BASE="/scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26"

print_usage() {
  cat <<EOF
Usage:
  Experiment mode:
    $0 <experiment_name> [local_dest_file_or_dir]

    Finds <experiment_name> anywhere under ${REMOTE_BASE} (nested allowed), then fetches:
      <found_dir>/ckpts/simclr/last.pt
    Fallback path:
      <found_dir>/cpkts/simclr/last.pt
    Default local destination: ./ckpts/simclr/<experiment_name>/last.pt

  Raw rsync mode:
    $0 <remote_path> <local_path> [--exclude-ext ext1,ext2,...]

Examples:
  $0 my_exp_name                          # saves to ./ckpts/simclr/my_exp_name/last.pt
  $0 my_exp_name ./runs/my_exp_last.pt    # saves to specified file
  $0 /home/kostouso/project/ ./backup/    # rsync a directory
  $0 /home/kostouso/project/ ./backup/ --exclude-ext .tmp,.log
EOF
}

EXCLUDE_EXTS=""

raw_rsync_mode=false

# Determine mode:
# - If first argument starts with '/', assume raw rsync mode (explicit remote path)
# - If only one argument provided and it's not starting with '/', treat as experiment mode
# - If two arguments provided and the first does not start with '/', treat as experiment mode with explicit local dest
if [[ $# -eq 0 ]]; then
  print_usage
  exit 1
fi

if [[ $1 == /* ]]; then
  raw_rsync_mode=true
fi

# Parse optional --exclude-ext argument for raw mode
if [[ "$raw_rsync_mode" == true ]]; then
  if [[ $# -lt 2 ]]; then
    print_usage
    exit 1
  fi
  REMOTE_PATH="$1"
  LOCAL_PATH="$2"
  if [[ $# -ge 3 && "$3" == "--exclude-ext" ]]; then
    if [[ $# -lt 4 ]]; then
      echo "Error: --exclude-ext requires an argument"
      exit 1
    fi
    EXCLUDE_EXTS="$4"
  fi
else
  # Experiment mode
  EXP_NAME="$1"
  # Default local destination mirrors remote subdirs
  DEFAULT_LOCAL_DIR="./ckpts/simclr/${EXP_NAME}"
  DEFAULT_LOCAL_FILE="${DEFAULT_LOCAL_DIR}/best_linear_probe.pt"
  if [[ $# -ge 2 ]]; then
    LOCAL_PATH="$2"
  else
    LOCAL_PATH="$DEFAULT_LOCAL_FILE"
  fi
  # Locate the experiment directory anywhere under REMOTE_BASE
  # Collect up to 5 matches to report if ambiguous
  mapfile -t FOUND_DIRS < <(ssh -F "$SSH_CONFIG" -i "$SSH_KEY" "$REMOTE_USER_HOST" \
    "find \"${REMOTE_BASE}\" -type d -name \"${EXP_NAME}\" 2>/dev/null | head -n 5")

  if [[ ${#FOUND_DIRS[@]} -eq 0 ]]; then
    echo "Error: Could not find a directory named '${EXP_NAME}' under ${REMOTE_BASE}"
    exit 1
  fi

  REMOTE_DIR="${FOUND_DIRS[0]}"
  if [[ ${#FOUND_DIRS[@]} -gt 1 ]]; then
    echo "Warning: Multiple matches found for '${EXP_NAME}'. Using: ${REMOTE_DIR}"
    printf "Other matches:\n"
    for d in "${FOUND_DIRS[@]:1}"; do
      printf "  %s\n" "$d"
    done
  fi

  # Candidate remote paths within the found directory (ckpts first, then cpkts fallback)
  REMOTE_PATH_CKPTS="${REMOTE_DIR}/ckpts/simclr/best_linear_probe.pt"
  REMOTE_PATH_CPKTS="${REMOTE_DIR}/cpkts/simclr/best_linear_probe.pt"
  REMOTE_PATH="${REMOTE_PATH_CKPTS}"
  # Check existence on remote and fallback if needed
  if ! ssh -F "$SSH_CONFIG" -i "$SSH_KEY" "$REMOTE_USER_HOST" test -f "$REMOTE_PATH_CKPTS"; then
    if ssh -F "$SSH_CONFIG" -i "$SSH_KEY" "$REMOTE_USER_HOST" test -f "$REMOTE_PATH_CPKTS"; then
      REMOTE_PATH="$REMOTE_PATH_CPKTS"
    else
      echo "Error: Neither path exists on remote within '${REMOTE_DIR}':"
      echo "  ${REMOTE_PATH_CKPTS}"
      echo "  ${REMOTE_PATH_CPKTS}"
      exit 1
    fi
  fi
  # Ensure local directory exists
  if [[ "${LOCAL_PATH}" == */ ]]; then
    LOCAL_DIR="${LOCAL_PATH%/}"
  elif [[ -d "${LOCAL_PATH}" ]]; then
    LOCAL_DIR="${LOCAL_PATH}"
  else
    LOCAL_DIR=$(dirname "$LOCAL_PATH")
  fi
  mkdir -p "$LOCAL_DIR"
fi

# rsync options:
#  -a : archive (preserves perms/times, recursive)
#  -v : verbose
#  -z : compress during transfer
#  --progress : show progress
#  --partial --append-verify : better for interrupted large files
RSYNC_OPTS=(-avz --progress --partial --append-verify)


EXCLUDE_OPTS=()
if [[ -n "${EXCLUDE_EXTS}" ]]; then
  IFS=',' read -r -a exts <<< "${EXCLUDE_EXTS}"
  for ext in "${exts[@]}"; do
    ext="${ext#.}"
    [[ -n "$ext" ]] && EXCLUDE_OPTS+=(--exclude="*.${ext}")
  done
fi



# Tell rsync to use the same SSH options you use for ssh
RSYNC_SSH=(ssh -F "$SSH_CONFIG" -i "$SSH_KEY")

rsync "${RSYNC_OPTS[@]}" "${EXCLUDE_OPTS[@]}" -e "${RSYNC_SSH[*]}" \
  "${REMOTE_USER_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"