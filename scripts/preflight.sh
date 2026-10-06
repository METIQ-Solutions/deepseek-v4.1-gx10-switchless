#!/usr/bin/env bash
#
# Read-only preflight for the whole four-node switchless deployment.
#
# Contacts every rank, stages nothing permanently, changes nothing on the
# hosts, and exits non-zero if any rank is not ready to serve. Run this before
# start.sh, and again after any reboot, cabling change or package update.
#
# Usage:
#   scripts/preflight.sh              # all ranks
#   scripts/preflight.sh --rank 2     # one rank
#
# Remote contact requires that you have already recorded the hosts in
# known_hosts; the scripts never disable host-key checking (see README).

set -euo pipefail

SCRIPT_NAME=preflight
# shellcheck source=./lib.sh
. "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

readonly PREFLIGHT_HELPER="${LIB_DIR}/host-preflight.sh"
readonly GID_HELPER="${LIB_DIR}/discover-roce-gids.py"
readonly REMOTE_GID_HELPER="/tmp/deepseek-v41-discover-roce-gids.py"

ranks=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --rank) [ "$#" -ge 2 ] || die "--rank requires a value"; ranks+=("$2"); shift 2 ;;
    -h|--help) sed -n '3,16p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

load_topology
validate_topology
if [ "${#ranks[@]}" -eq 0 ]; then
  for ((rank = 0; rank < NODE_COUNT; rank++)); do ranks+=("${rank}"); done
fi

log "preflight for ${NODE_COUNT} ranks (cluster ${CLUSTER_NAME})"

overall=0
for rank in "${ranks[@]}"; do
  name="$(rank_name "${rank}")"
  log "--- rank ${rank} (${name})"

  if ! rank_run "${rank}" true 2>/dev/null; then
    warn "rank ${rank} (${name}) is not reachable over SSH"
    overall=1
    continue
  fi

  # Stage the GID helper read-only for the duration of the check. It is a
  # pure reader of sysfs, so leaving it in /tmp is not a mutation of state.
  rank_run "${rank}" "cat > ${REMOTE_GID_HELPER}" < "${GID_HELPER}"

  preflight_args=(
    --rank "${rank}"
    --control-ip "$(rank_ctrl_ip "${rank}")"
    --control-iface "$(rank_iface "${rank}")"
    --hca "${NCCL_IB_HCA}"
    --gid-index "${NCCL_GID_INDEX:-}"
    --fabric "$(rank_fabric "${rank}")"
    --model-root "${MODEL_ROOT}"
    --engram-dir "${WORKLOAD_ROOT}/patches"
    --image "${IMAGE}"
    --headroom-gib "${STARTUP_HEADROOM_GIB:-16}"
    --memory-slice "vllm-${CLUSTER_NAME}.slice"
    --gid-helper "${REMOTE_GID_HELPER}"
  )
  [ "${rank}" -eq "${HEAD_RANK}" ] && preflight_args+=(--expect-api-port "${API_PORT}")

  if rank_run "${rank}" "bash -s -- $(printf '%q ' "${preflight_args[@]}")" < "${PREFLIGHT_HELPER}"; then
    :
  else
    warn "rank ${rank} (${name}) is not ready"
    overall=1
  fi

  rank_run "${rank}" "rm -f ${REMOTE_GID_HELPER}" || true
done

if [ "${overall}" -ne 0 ]; then
  die "preflight failed; fix the reported checks (they name the script to run) before start.sh"
fi
log "PREFLIGHT_PASS cluster=${CLUSTER_NAME} ranks=${NODE_COUNT}"
