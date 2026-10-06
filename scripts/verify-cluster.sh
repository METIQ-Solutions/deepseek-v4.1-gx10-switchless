#!/usr/bin/env bash
#
# Prove the switchless deployment is what it claims to be.
#
# Four independent proofs, in order:
#
#   A. the running serving container loaded the ring-only NCCL build and used
#      native IB, with no Socket or Mesh tensor fallback;
#   B. a MODEL-FREE four-rank collective completes over the ring, so the
#      transport is proven independently of the model;
#   C. tensor traffic actually crossed every physical ring edge (per-HCA byte
#      counters around a real 32K prefill), because a collective can otherwise
#      succeed over a path the ring cannot carry;
#   D. functional acceptance: model identity, real generation, tool calls and
#      reasoning levels.
#
# Run after start.sh. Read-only with respect to the deployment (B/C do create
# one short-lived helper container, which is removed afterwards).
#
# Usage: scripts/verify-cluster.sh [--skip-collective] [--skip-traffic]

set -euo pipefail

SCRIPT_NAME=verify
# shellcheck source=./lib.sh
. "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

skip_collective=0
skip_traffic=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip-collective) skip_collective=1; shift ;;
    --skip-traffic) skip_traffic=1; shift ;;
    -h|--help) sed -n '3,20p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

load_topology
validate_topology

readonly HEAD_IP="$(rank_ctrl_ip "${HEAD_RANK}")"
readonly BASE_URL="http://${HEAD_IP}:${API_PORT}"
readonly COLLECTIVE_PORT="$((MASTER_PORT + 100))"
readonly COLLECTIVE_NAME="dsv41-collective"
readonly REMOTE_SCRATCH="/tmp/deepseek-v4.1-verify"
readonly MIN_EDGE_BYTES="${MIN_EDGE_BYTES:-1048576}"

work_dir="$(mktemp -d --tmpdir v41-verify.XXXXXX)"
trap 'rm -rf -- "${work_dir}"' EXIT

# ---------------------------------------------------------------------------
# A. transport evidence in the running serving container
# ---------------------------------------------------------------------------
log "A: transport evidence in the running serving container"

for rank in "${HEAD_RANK}" "$(( (HEAD_RANK + 1) % NODE_COUNT ))"; do
  logs="$(rank_run "${rank}" "docker logs --tail 4000 ${CONTAINER_NAME:-deepseek-v4.1-vllm} 2>&1" || true)"
  [ -n "${logs}" ] || die "rank ${rank}: no container logs available"

  printf '%s' "${logs}" | grep -q 'SWITCHLESS/HARDENED' \
    || die "rank ${rank}: the ring-only NCCL build did not report its hardened/skip markers"
  printf '%s' "${logs}" | grep -Eq 'Using network IB|NET/IB' \
    || die "rank ${rank}: no native IB transport evidence in the container logs"
  if printf '%s' "${logs}" | grep -Eq 'via NET/Socket|NET/Mesh'; then
    die "rank ${rank}: tensor transport fell back to Socket or Mesh, which a switchless ring cannot carry"
  fi

  # --- Tree-override fingerprints (see docs/nccl-contract.md) ---------------
  # A process that rewrote NCCL_ALGO to Tree leaves these behind even though the
  # configured values are Ring/4 channels. Catching them here turns a later
  # "no transport for recv peer" failure into an immediate, precise diagnosis.
  if printf '%s' "${logs}" | grep -Eq 'coll channels'; then
    coll_channels="$(printf '%s' "${logs}" | grep -oE '[0-9]+ coll channels' | head -n1 | awk '{print $1}')"
    if [ "${coll_channels:-0}" -lt 4 ]; then
      die "rank ${rank}: only ${coll_channels} collectives channel(s); the qualified ring uses 4 (Tree-forcing override?)"
    fi
  fi
  if printf '%s' "${logs}" | grep -q 'Invalid NCCL_NTHREADS'; then
    die "rank ${rank}: NCCL_NTHREADS was set by the process; the switchless contract leaves it unset (Tree-forcing override?)"
  fi
  log "rank ${rank}: ring-only NCCL active, native IB, no Socket/Mesh fallback"
done

# The effective environment of the running head container is the only place the
# batch-invariance landmine can be observed directly, because it is applied
# in-process after the compose environment is read.
head_env="$(rank_run "${HEAD_RANK}" "docker exec ${CONTAINER_NAME:-deepseek-v4.1-vllm} printenv VLLM_BATCH_INVARIANT" 2>/dev/null || true)"
case "${head_env:-0}" in
  0|"") log "head: VLLM_BATCH_INVARIANT is off (ring contract intact)" ;;
  *) die "head: VLLM_BATCH_INVARIANT=${head_env}; batch invariance rewrites NCCL_ALGO to Tree and this fabric cannot carry it" ;;
esac

# ---------------------------------------------------------------------------
# B. model-free four-rank collective
# ---------------------------------------------------------------------------
if [ "${skip_collective}" -eq 0 ]; then
  log "B: model-free four-rank NCCL collective over the ring"

  for ((rank = 0; rank < NODE_COUNT; rank++)); do
    rank_run "${rank}" "mkdir -p '${REMOTE_SCRATCH}' && cat > '${REMOTE_SCRATCH}/nccl-collective.py'" \
      < "${LIB_DIR}/nccl-collective.py"
    rank_run "${rank}" "docker rm -f '${COLLECTIVE_NAME}' >/dev/null 2>&1 || true"
  done

  for ((rank = 0; rank < NODE_COUNT; rank++)); do
    # Detached so all four ranks rendezvous; docker wait/logs then give the
    # exit code and the transport evidence without an SSH session holding the
    # collective open.
    rank_run "${rank}" "docker run -d --name '${COLLECTIVE_NAME}' \
      --network host --ipc host --gpus all --device /dev/infiniband:/dev/infiniband \
      --env-file '${WORKLOAD_ROOT}/config/vllm.env' \
      --env-file '${WORKLOAD_ROOT}/compose/.env' \
      -e RANK='${rank}' -e WORLD_SIZE='${NODE_COUNT}' \
      -e MASTER_ADDR='${HEAD_IP}' -e MASTER_PORT='${COLLECTIVE_PORT}' \
      -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH \
      -v '${REMOTE_SCRATCH}/nccl-collective.py:/nccl-collective.py:ro' \
      '${IMAGE}' python3 /nccl-collective.py"
  done

  collective_failed=0
  for ((rank = 0; rank < NODE_COUNT; rank++)); do
    if ! rank_run "${rank}" "docker wait '${COLLECTIVE_NAME}'" | grep -qx 0; then
      warn "collective failed on rank ${rank}"
      rank_run "${rank}" "docker logs '${COLLECTIVE_NAME}' 2>&1 | tail -n 40" || true
      collective_failed=1
    fi
    rank_run "${rank}" "docker logs '${COLLECTIVE_NAME}' 2>&1" > "${work_dir}/collective-rank${rank}.log" || true
    rank_run "${rank}" "docker rm -f '${COLLECTIVE_NAME}' >/dev/null 2>&1 || true"
  done
  [ "${collective_failed}" -eq 0 ] || die "the model-free four-rank collective did not succeed on every rank"

  cat "${work_dir}"/collective-rank*.log | grep -q 'COLLECTIVE_OK' \
    || die "the collective reported no success line"
  cat "${work_dir}"/collective-rank*.log | grep -Eq 'Using network IB|NET/IB' \
    || die "the collective used no native IB transport"
  if cat "${work_dir}"/collective-rank*.log | grep -Eq 'via NET/Socket|NET/Mesh'; then
    die "the collective fell back to Socket or Mesh"
  fi
  log "all four ranks completed an all-reduce over native IB"
fi

# ---------------------------------------------------------------------------
# C. per-edge tensor traffic
# ---------------------------------------------------------------------------
if [ "${skip_traffic}" -eq 0 ]; then
  log "C: per-edge tensor traffic around a real prefill"

  for ((rank = 0; rank < NODE_COUNT; rank++)); do
    rank_run "${rank}" "cat > '${REMOTE_SCRATCH}/collect-fabric-metrics.py'" < "${LIB_DIR}/collect-fabric-metrics.py"
    interfaces_json="$(python3 "${LIB_DIR}/validate-topology.py" "${TOPOLOGY_FILE}" --interfaces "${rank}")"
    rank_run "${rank}" "python3 '${REMOTE_SCRATCH}/collect-fabric-metrics.py' '${interfaces_json}' '$(rank_iface "${rank}")'" \
      > "${work_dir}/before-rank${rank}.json"
  done

  rank_run "${HEAD_RANK}" "cat > '${REMOTE_SCRATCH}/benchmark.py'" < "${LIB_DIR}/../benchmarks/scripts/benchmark.py"
  rank_run "${HEAD_RANK}" "python3 '${REMOTE_SCRATCH}/benchmark.py' --endpoint '${BASE_URL}' \
    --json-out '${REMOTE_SCRATCH}/traffic.json' --context-token 32768 --samples 1 --prefill-only --max-tokens 64" \
    >/dev/null

  for ((rank = 0; rank < NODE_COUNT; rank++)); do
    interfaces_json="$(python3 "${LIB_DIR}/validate-topology.py" "${TOPOLOGY_FILE}" --interfaces "${rank}")"
    rank_run "${rank}" "python3 '${REMOTE_SCRATCH}/collect-fabric-metrics.py' '${interfaces_json}' '$(rank_iface "${rank}")'" \
      > "${work_dir}/after-rank${rank}.json"
  done

  python3 "${LIB_DIR}/check-edge-deltas.py" \
    --topology "${TOPOLOGY_FILE}" \
    --samples-dir "${work_dir}" \
    --minimum-bytes "${MIN_EDGE_BYTES}"

  for ((rank = 0; rank < NODE_COUNT; rank++)); do
    rank_run "${rank}" "rm -rf '${REMOTE_SCRATCH}'" || true
  done
fi

# ---------------------------------------------------------------------------
# D. functional acceptance
# ---------------------------------------------------------------------------
log "D: functional acceptance against ${BASE_URL}"
acceptance_args=(--endpoint "${BASE_URL}" --model "$(sed -n 's/^SERVED_MODEL_NAME=//p' "${ENV_FILE}")" --with-tool-call --with-speculative)
python3 "${LIB_DIR}/acceptance.py" "${acceptance_args[@]}"

log "VERIFY_PASS cluster=${CLUSTER_NAME} transport=ring-only-inb edges=${NODE_COUNT} acceptance=ok"
