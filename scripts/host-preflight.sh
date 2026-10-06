#!/usr/bin/env bash
#
# Read-only preflight for ONE host. Makes no changes.
#
# Invoked by scripts/preflight.sh over SSH, but it can also be run directly on
# a node. Prints one `CHECK <name> <PASS|FAIL> <detail>` line per check and
# exits non-zero if any check failed.
#
# Usage (on the host):
#   bash host-preflight.sh --rank N --control-ip IP --control-iface IFACE \
#     --hca "=rocep1s0f0:1,rocep1s0f1:1" --gid-index 3 \
#     --fabric "iface:addr/prefix ..." --model-root DIR --engram-dir DIR \
#     --image TAG --headroom-gib 16 --min-mtu 9000 \
#     [--gid-helper /path/to/discover-roce-gids.py] [--expect-api-port 8889]

set -uo pipefail

rank=""; control_ip=""; control_iface=""; hca=""; gid_index=""
fabric=""; model_root=""; engram_dir=""; image=""; headroom_gib="16"
min_mtu="9000"; gid_helper=""; expect_api_port=""; memory_slice=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --rank) rank="$2"; shift 2 ;;
    --control-ip) control_ip="$2"; shift 2 ;;
    --control-iface) control_iface="$2"; shift 2 ;;
    --hca) hca="$2"; shift 2 ;;
    --gid-index) gid_index="$2"; shift 2 ;;
    --fabric) fabric="$2"; shift 2 ;;
    --model-root) model_root="$2"; shift 2 ;;
    --engram-dir) engram_dir="$2"; shift 2 ;;
    --image) image="$2"; shift 2 ;;
    --headroom-gib) headroom_gib="$2"; shift 2 ;;
    --min-mtu) min_mtu="$2"; shift 2 ;;
    --memory-slice) memory_slice="$2"; shift 2 ;;
    --gid-helper) gid_helper="$2"; shift 2 ;;
    --expect-api-port) expect_api_port="$2"; shift 2 ;;
    *) echo "ERROR unknown argument: $1" >&2; exit 2 ;;
  esac
done

failures=0
pass() { printf 'CHECK %s PASS %s\n' "$1" "$2"; }
fail() { printf 'CHECK %s FAIL %s\n' "$1" "$2"; failures=$((failures + 1)); }

# --- tooling ----------------------------------------------------------------
if command -v docker >/dev/null 2>&1; then
  pass docker "$(docker --version 2>/dev/null | head -n1)"
else
  fail docker "docker is not installed"
fi

# --- control plane ----------------------------------------------------------
if ip -4 -o address show dev "${control_iface}" 2>/dev/null | grep -q " ${control_ip}/"; then
  pass control_address "${control_ip} on ${control_iface}"
else
  fail control_address "${control_ip} is not present on ${control_iface}"
fi

# --- fabric: address, MTU and absence of a default route --------------------
fabric_ok=1
fabric_detail=""
read -r -a fabric_entries <<< "${fabric}"
if [ "${#fabric_entries[@]}" -ne 4 ]; then
  fabric_ok=0
  fabric_detail="expected 4 fabric entries, got ${#fabric_entries[@]}"
fi
declare -a fabric_ifaces=()
for entry in "${fabric_entries[@]}"; do
  iface="${entry%%:*}"
  address="${entry#*:}"
  fabric_ifaces+=("${iface}")
  if ! ip -o link show dev "${iface}" >/dev/null 2>&1; then
    fabric_ok=0; fabric_detail="${fabric_detail} ${iface}:missing"
    continue
  fi
  if ! ip -4 -o address show dev "${iface}" 2>/dev/null | grep -q " ${address} "; then
    fabric_ok=0; fabric_detail="${fabric_detail} ${iface}:address-mismatch(${address})"
  fi
  mtu="$(cat "/sys/class/net/${iface}/mtu" 2>/dev/null || echo 0)"
  if [ "${mtu}" -lt "${min_mtu}" ]; then
    fabric_ok=0; fabric_detail="${fabric_detail} ${iface}:mtu=${mtu}"
  fi
  # A default route on a point-to-point ring lane would let stray traffic leave
  # the ring and silently change the transport this runtime was validated with.
  if ip -4 route show default dev "${iface}" 2>/dev/null | grep -q .; then
    fabric_ok=0; fabric_detail="${fabric_detail} ${iface}:default-route"
  fi
done
if [ "${#fabric_ifaces[@]}" -ne 0 ] && printf '%s\n' "${fabric_ifaces[@]}" | grep -qxF "${control_iface}"; then
  fabric_ok=0; fabric_detail="${fabric_detail} control-interface-used-as-fabric-lane"
fi
if [ "${fabric_ok}" -eq 1 ]; then
  pass fabric "4 lanes bound, MTU>=${min_mtu}, no default route, control plane separate"
else
  fail fabric "${fabric_detail}"
fi

# --- RDMA devices -----------------------------------------------------------
if [ -d /dev/infiniband ]; then
  pass infiniband "/dev/infiniband present"
else
  fail infiniband "/dev/infiniband is missing; the container cannot use RDMA"
fi

# --- RoCE GID index ---------------------------------------------------------
if [ -n "${gid_helper}" ] && [ -f "${gid_helper}" ]; then
  declare -A gid_addresses=()
  for entry in "${fabric_entries[@]}"; do
    iface="${entry%%:*}"; address="${entry#*:}"
    # Only the two lanes NCCL is allowed to select are relevant here: the
    # helper derives HCA names from interface names, so pass all four and let
    # it pick the selected ones.
    gid_addresses["${iface}"]="${address}"
  done
  json="{"
  first=1
  for iface in "${!gid_addresses[@]}"; do
    [ "${first}" -eq 1 ] || json="${json},"
    json="${json}\"${iface}\":\"${gid_addresses[${iface}]}\""
    first=0
  done
  json="${json}}"
  if gid_output="$(python3 "${gid_helper}" "${json}" "${hca}" 2>&1)"; then
    observed="$(printf '%s' "${gid_output}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["index"])')"
    if [ -n "${gid_index}" ] && [ "${observed}" != "${gid_index}" ]; then
      fail roce_gid "observed common index ${observed}, topology declares ${gid_index}"
    else
      pass roce_gid "common IPv4 RoCE-v2 GID index ${observed}"
    fi
  else
    fail roce_gid "${gid_output}"
  fi
else
  fail roce_gid "GID helper not provided; run scripts/preflight.sh so it is staged automatically"
fi

# --- ring-only NCCL library -------------------------------------------------
nccl_library="/opt/switchless-nccl/libnccl.so.2.30.7"
nccl_sha="78cb83871792ec57d763d142e4cae26fc754ae284bcc81dcb2a7d50e17d4fa57"
if [ -f "${nccl_library}" ] && [ "$(sha256sum "${nccl_library}" | cut -d' ' -f1)" = "${nccl_sha}" ]; then
  pass switchless_nccl "${nccl_library}";
else
  fail switchless_nccl "${nccl_library} missing or hash mismatch; run scripts/install-switchless-nccl.sh"
fi

# --- image ------------------------------------------------------------------
if [ -n "${image}" ] && command -v docker >/dev/null 2>&1; then
  if docker image inspect "${image}" >/dev/null 2>&1; then
    pass image "${image} present"
  else
    fail image "${image} is not present in the local docker store"
  fi
fi

# --- nothing already running ------------------------------------------------
if command -v docker >/dev/null 2>&1; then
  if [ "$(docker inspect --format '{{.State.Running}}' deepseek-v4.1-vllm 2>/dev/null || echo false)" = "true" ]; then
    fail container "deepseek-v4.1-vllm is already running; stop it first"
  else
    pass container "no running deepseek-v4.1-vllm"
  fi
fi

# --- model ------------------------------------------------------------------
if [ -d "${model_root}" ]; then
  shard_count="$(find "${model_root}" -maxdepth 1 -name '*.safetensors' | wc -l)"
  if [ "${shard_count}" -ge 48 ]; then
    pass model "${model_root} present with ${shard_count} safetensors shards"
  else
    fail model "${model_root} has only ${shard_count} safetensors shards, expected 48"
  fi
else
  fail model "${model_root} does not exist"
fi

# --- staged Engram / SM12x patches -----------------------------------------
expected_patches="engram.py model_state.py weight_utils.py attention.py flashinfer_sparse.py sparse_swa.py sparse_attn_indexer.py"
missing_patches=""
for name in ${expected_patches}; do
  [ -f "${engram_dir}/${name}" ] || missing_patches="${missing_patches} ${name}"
done
if [ -z "${missing_patches}" ]; then
  pass engram_patches "7 files staged in ${engram_dir}"
else
  fail engram_patches "missing:${missing_patches}; run scripts/fetch-engram-patches.sh"
fi

# --- host memory headroom ---------------------------------------------------
available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
needed_kib=$((headroom_gib * 1024 * 1024))
if [ "${available_kib}" -ge "${needed_kib}" ]; then
  pass headroom "$((available_kib / 1024 / 1024)) GiB available (>= ${headroom_gib} GiB)"
else
  fail headroom "$((available_kib / 1024 / 1024)) GiB available, need ${headroom_gib} GiB before starting"
fi

# --- memory boundary --------------------------------------------------------
slice_state="$(systemctl show --property=ActiveState --value "${memory_slice}" 2>/dev/null || echo unknown)"
if [ "${slice_state}" = "active" ]; then
  pass memory_boundary "${memory_slice} active"
else
  fail memory_boundary "${memory_slice} is ${slice_state}; run scripts/install-memory-boundary.sh"
fi

# --- head-only checks -------------------------------------------------------
if [ -n "${expect_api_port}" ] && [ "${rank}" = "0" ]; then
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${expect_api_port}\$"; then
    fail api_port "port ${expect_api_port} is already in use on the head"
  else
    pass api_port "port ${expect_api_port} free on the head"
  fi
fi

if [ "${failures}" -ne 0 ]; then
  printf 'HOST_PREFLIGHT_FAIL rank=%s failures=%s\n' "${rank}" "${failures}" >&2
  exit 1
fi
printf 'HOST_PREFLIGHT_PASS rank=%s\n' "${rank}"
