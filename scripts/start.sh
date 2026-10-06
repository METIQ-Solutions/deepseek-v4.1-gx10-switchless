#!/usr/bin/env bash
#
# Start the four-rank switchless DeepSeek V4.1 deployment.
#
# Order matters and is deliberate:
#
#   1. converge every host (ring-only NCCL, Engram patches, memory boundary);
#   2. fence any previous generation so no stale rank survives;
#   3. start the WORKERS first, so they are already waiting in the rendezvous
#      path when the head arrives;
#   4. start the head and wait for its API to become healthy;
#   5. wait for every rank container to be running;
#   6. run the bounded inference canary, which proves distributed generation
#      works rather than only that the API answers.
#
# Nothing here is automatic on boot by design: a distributed generation must be
# established as a unit, never reconstructed rank-by-rank after a reboot.
#
# Usage: scripts/start.sh [--skip-converge] [--no-canary]

set -euo pipefail

SCRIPT_NAME=start
# shellcheck source=./lib.sh
. "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

skip_converge=0
run_canary=1
while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip-converge) skip_converge=1; shift ;;
    --no-canary) run_canary=0; shift ;;
    -h|--help) sed -n '3,22p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

load_topology
validate_topology

readonly PROJECT="${COMPOSE_PROJECT:-deepseek-v4-1}"
readonly CONTAINER="${CONTAINER_NAME:-deepseek-v4.1-vllm}"
readonly SLICE="vllm-${CLUSTER_NAME}.slice"
readonly REMOTE_CONFIG="${WORKLOAD_ROOT}/config"
readonly REMOTE_COMPOSE="${WORKLOAD_ROOT}/compose"
readonly STARTUP_TIMEOUT="${STARTUP_TIMEOUT_SECONDS:-3600}"

log "starting ${CLUSTER_NAME}: ${NODE_COUNT} ranks, TP${CLUSTER_TP_SIZE}, head rank ${HEAD_RANK}"

# --- 1. converge every host -------------------------------------------------
if [ "${skip_converge}" -eq 0 ]; then
  for ((rank = 0; rank < NODE_COUNT; rank++)); do
    log "converging rank ${rank} ($(rank_name "${rank}"))"
    rank_run "${rank}" "mkdir -p '${WORKLOAD_ROOT}' '${REMOTE_CONFIG}' '${REMOTE_COMPOSE}' '${CACHE_ROOT}' '${WORKLOAD_ROOT}/tmp'"
    rank_run "${rank}" "bash -s" < "${LIB_DIR}/install-switchless-nccl.sh"
    rank_run "${rank}" "bash -s -- --dest '${WORKLOAD_ROOT}/patches'" < "${LIB_DIR}/fetch-engram-patches.sh"
    rank_run "${rank}" "bash -s -- --slice '${SLICE}' --high '${MEMORY_HIGH}' --max '${MEMORY_MAX}' --swap-max '${MEMORY_SWAP_MAX:-0}'" \
      < "${LIB_DIR}/install-memory-boundary.sh"
    # The compose file and the shared environment are rendered from this
    # repository, so every rank runs the exact revision you checked out.
    rank_run "${rank}" "cat > '${REMOTE_CONFIG}/vllm.env'" < "${ENV_FILE}"
    rank_run "${rank}" "cat > '${REMOTE_COMPOSE}/compose.yaml'" < "${COMPOSE_FILE}"
    printf '%s\n' "$(write_rank_env "${rank}")" | rank_run "${rank}" "cat > '${REMOTE_COMPOSE}/.env'"
  done
else
  log "skipping host convergence (--skip-converge)"
fi

# --- 2. fence any previous generation ---------------------------------------
for ((rank = 0; rank < NODE_COUNT; rank++)); do
  if [ "${rank}" -eq "${HEAD_RANK}" ]; then
    fence_rank "${rank}" yes "${SLICE}"
  else
    fence_rank "${rank}" no "${SLICE}"
  fi
done

compose_up() {
  local rank="$1"
  rank_run "${rank}" "docker compose --project-name '${PROJECT}' --env-file '${REMOTE_COMPOSE}/.env' -f '${REMOTE_COMPOSE}/compose.yaml' up -d"
}

# --- 3. workers first -------------------------------------------------------
for ((rank = 0; rank < NODE_COUNT; rank++)); do
  [ "${rank}" -eq "${HEAD_RANK}" ] && continue
  log "starting worker rank ${rank} ($(rank_name "${rank}"))"
  compose_up "${rank}"
done

for ((rank = 0; rank < NODE_COUNT; rank++)); do
  [ "${rank}" -eq "${HEAD_RANK}" ] && continue
  wait_for_container_running "${rank}" "${CONTAINER}" 900
done
log "all workers running; starting the head"

# --- 4. head ----------------------------------------------------------------
compose_up "${HEAD_RANK}"
wait_for_head_api "${HEAD_RANK}" "${STARTUP_TIMEOUT}"
wait_for_container_running "${HEAD_RANK}" "${CONTAINER}" 900

# --- 5. every rank running --------------------------------------------------
for ((rank = 0; rank < NODE_COUNT; rank++)); do
  wait_for_container_running "${rank}" "${CONTAINER}" 300
done

# --- 6. canary --------------------------------------------------------------
if [ "${run_canary}" -eq 1 ]; then
  log "running the bounded inference canary on the head"
  rank_run "${HEAD_RANK}" bash -s -- "http://$(rank_ctrl_ip "${HEAD_RANK}"):${API_PORT}" \
    "$(sed -n 's/^SERVED_MODEL_NAME=//p' "${ENV_FILE}")" < "${LIB_DIR}/canary.sh"
fi

log "START_PASS cluster=${CLUSTER_NAME} ranks=${NODE_COUNT} api=http://$(rank_ctrl_ip "${HEAD_RANK}"):${API_PORT}/v1"
log "next: scripts/verify-cluster.sh for the transport and functional proof"
