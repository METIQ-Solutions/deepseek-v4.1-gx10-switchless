#!/usr/bin/env bash
#
# Stop the four-rank deployment with a checked local fence.
#
# The head (rank 0) is fenced FIRST so the rendezvous endpoint is confirmed gone
# before the workers are stopped; stopping workers first would leave the head
# retrying against a peer that no longer exists. Each fence verifies the
# container it is about to stop actually belongs to this Compose project, stops
# only the resolved full container ID, and keeps all model, cache and volume
# data.
#
# Idempotent: stopping an already-idle estate reports success.
#
# Usage: scripts/stop.sh [--force]

set -euo pipefail

SCRIPT_NAME=stop
# shellcheck source=./lib.sh
. "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) sed -n '3,16p' "$0"; exit 0 ;;
    --force) shift ;;
    *) die "unknown argument: $1" ;;
  esac
done

load_topology
validate_topology

readonly SLICE="vllm-${CLUSTER_NAME}.slice"
readonly CONTAINER="${CONTAINER_NAME:-deepseek-v4.1-vllm}"

log "stopping ${CLUSTER_NAME}"

log "fencing the head (rank ${HEAD_RANK})"
fence_rank "${HEAD_RANK}" yes "${SLICE}"

for ((rank = 0; rank < NODE_COUNT; rank++)); do
  [ "${rank}" -eq "${HEAD_RANK}" ] && continue
  log "fencing worker rank ${rank} ($(rank_name "${rank}"))"
  fence_rank "${rank}" no "${SLICE}"
done

# Confirm the terminal state rather than trusting the fence exit codes alone.
for ((rank = 0; rank < NODE_COUNT; rank++)); do
  state="$(rank_run "${rank}" "docker inspect --format '{{.State.Running}}' '${CONTAINER}' 2>/dev/null || echo absent")"
  case "${state}" in
    true) die "rank ${rank} still reports the container running after the fence" ;;
    absent|false) ;;
    *) die "rank ${rank} returned an unexpected container state: ${state}" ;;
  esac
done

if rank_run "${HEAD_RANK}" "ss -ltn 2>/dev/null | awk '{print \$4}' | grep -q ':${MASTER_PORT}\$'"; then
  die "the rendezvous port ${MASTER_PORT} is still listening on the head"
fi

log "STOP_PASS cluster=${CLUSTER_NAME} containers stopped rendezvous absent data preserved"
