#!/usr/bin/env bash
#
# Create and activate the host-memory boundary for the managed vLLM rank.
#
# The container is placed BELOW this systemd slice, so the limit covers the
# container, its pinned host memory and the tested CUDA/unified-memory charges
# while leaving an explicit reserve for the operating system and management
# plane. Swap is never used: reclaiming to swap on a unified-memory node turns
# a slow path into a failure. See docs/safety.md for the derivation.
#
# Idempotent, and safe to run while nothing is serving.
#
# Usage: sudo bash install-memory-boundary.sh --slice NAME --high 104G --max 112G

set -euo pipefail

slice=""
high=""
maximum=""
swap_max="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --slice)    [ "$#" -ge 2 ] || { echo "ERROR --slice requires a value" >&2; exit 2; }; slice="$2"; shift 2 ;;
    --high)     [ "$#" -ge 2 ] || { echo "ERROR --high requires a value" >&2; exit 2; }; high="$2"; shift 2 ;;
    --max)      [ "$#" -ge 2 ] || { echo "ERROR --max requires a value" >&2; exit 2; }; maximum="$2"; shift 2 ;;
    --swap-max) [ "$#" -ge 2 ] || { echo "ERROR --swap-max requires a value" >&2; exit 2; }; swap_max="$2"; shift 2 ;;
    -h|--help)  sed -n '3,15p' "$0"; exit 0 ;;
    *) echo "ERROR unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "${slice}" ] || { echo "ERROR --slice is required" >&2; exit 2; }
[ -n "${high}" ] || { echo "ERROR --high is required" >&2; exit 2; }
[ -n "${maximum}" ] || { echo "ERROR --max is required" >&2; exit 2; }

case "${slice}" in
  *.slice) ;;
  *) echo "ERROR --slice must end in .slice (got ${slice})" >&2; exit 2 ;;
esac
if [ "${high}" = "${maximum}" ]; then
  echo "ERROR --high and --max must differ: an equal pair removes the early pressure boundary" >&2
  exit 2
fi

if [ "$(id -u)" -ne 0 ]; then
  echo "ERROR must run as root" >&2
  exit 1
fi

unit="/etc/systemd/system/${slice}"
if [ ! -f "${unit}" ] || ! grep -qF "MemoryMax=${maximum}" "${unit}"; then
  cat > "${unit}" <<EOF
# Managed vLLM host-memory boundary. Written by install-memory-boundary.sh.
#
# MemoryHigh is the earlier pressure/throttling boundary; MemoryMax is the hard
# workload ceiling; MemorySwapMax keeps reclaim from consuming host swap.
[Unit]
Description=Memory boundary for the DeepSeek V4.1 vLLM rank

[Slice]
MemoryHigh=${high}
MemoryMax=${maximum}
MemorySwapMax=${swap_max}
EOF
  systemctl daemon-reload
fi

systemctl start "${slice}"
state="$(systemctl show --property=ActiveState --value "${slice}")"
[ "${state}" = "active" ] || { echo "ERROR ${slice} is ${state}, expected active" >&2; exit 1; }

echo "MEMORY_BOUNDARY_OK slice=${slice} high=${high} max=${maximum} swap_max=${swap_max} state=${state}"
