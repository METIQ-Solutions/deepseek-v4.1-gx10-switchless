#!/usr/bin/env bash
#
# Stage the disk-backed Engram and GB10/SM12x runtime patches on THIS host.
#
# Runs on the target node (also piped over SSH by the lifecycle scripts). The
# seven files come from a pinned upstream revision and are verified
# individually; nothing is redistributed by this repository.
#
# One local composition step is required: the upstream `engram.py` imports
# `gather_engram_hashes` from the day-0 runtime, and the disk-backed Engram path
# must refuse the shared-memory Engram-DP mode rather than silently returning
# local hashes. The shim is appended with explicit markers and the result is
# compiled before it is used.
#
# Usage: sudo bash fetch-engram-patches.sh [--dest DIR] [--force]

set -euo pipefail

readonly REVISION="592540c69853a8ce9285236ebfd6e54dfc83a013"
readonly BASE_URL="https://raw.githubusercontent.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/${REVISION}/patch"

# name:source-sha256
readonly FILES=(
  "engram.py:91aa095fc3ef108c3ca66072a04442717ce054e3389d36a34d2a0d1a9f3b0bbd"
  "model_state.py:8cf3ff42fa9993a6539b26dc5ff0ca525df13192ce9b266a0519a238bbdffbcd"
  "weight_utils.py:6ad34626eec89f69257d211e17c02191eeec180782cdd388169843e3b0000701"
  "attention.py:638c06f0f2cf1176a7dac5877aed328923df88ca3b97c50b40e45ee09d7fff7b"
  "flashinfer_sparse.py:ab7712f977aa59db6e357310f20e38e1a506170a61ebaa6a2495d59d8a07ccc3"
  "sparse_swa.py:356b4693d174ee5bd714275be96422d542bfd6f18139693ed4e3000b4e240f34"
  "sparse_attn_indexer.py:8a8a1502357e11204a8ce49f207c70cd2b888518a1cea7c49790adb57aeac517"
)

readonly SHIM_BEGIN="# BEGIN DSV41 GATHER ENGRAM HASHES"
readonly SHIM_END="# END DSV41 GATHER ENGRAM HASHES"

dest=""
force=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dest)  [ "$#" -ge 2 ] || { echo "ERROR --dest requires a value" >&2; exit 2; }; dest="$2"; shift 2 ;;
    --force) force=1; shift ;;
    -h|--help) sed -n '3,16p' "$0"; exit 0 ;;
    *) echo "ERROR unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "${dest}" ] || { echo "ERROR --dest is required (the host patch directory mounted by compose)" >&2; exit 2; }

if [ "$(id -u)" -ne 0 ]; then
  echo "ERROR must run as root" >&2
  exit 1
fi

command -v curl >/dev/null 2>&1 || { echo "ERROR curl is required" >&2; exit 1; }

install -d -o root -g root -m 0755 "${dest}"

fetch_verified() {
  local name="$1" expected="$2" target="$3"
  if [ "${force}" -eq 0 ] && [ -f "${target}" ] \
     && [ "$(sha256sum "${target}" | cut -d' ' -f1)" = "${expected}" ]; then
    return 0
  fi
  curl --fail --location --silent --show-error --retry 5 \
    --connect-timeout 15 --max-time 120 \
    --output "${target}.incoming" "${BASE_URL}/${name}"
  printf '%s  %s\n' "${expected}" "${target}.incoming" | sha256sum -c - \
    || { echo "ERROR ${name} does not match the pinned upstream revision ${REVISION}" >&2; exit 1; }
  mv -- "${target}.incoming" "${target}"
}

for entry in "${FILES[@]}"; do
  name="${entry%%:*}"
  expected="${entry##*:}"
  fetch_verified "${name}" "${expected}" "${dest}/${name}"
  echo "ok ${name} (upstream revision ${REVISION})"
done

# --- local Engram-DP shim (the one non-upstream step) -----------------------
if ! grep -qF "${SHIM_BEGIN}" "${dest}/engram.py"; then
  cat >> "${dest}/engram.py" <<SHIM

${SHIM_BEGIN}
def gather_engram_hashes(
    hash_ids: torch.Tensor, *, dp_shared_memory: bool = False
) -> torch.Tensor:
    """Return local Engram hashes for the single-Engram-DP deployment."""
    if dp_shared_memory:
        raise NotImplementedError("disk-backed Engram does not support Engram DP")
    return hash_ids
${SHIM_END}
SHIM
fi

grep -qF "${SHIM_END}" "${dest}/engram.py" \
  || { echo "ERROR the Engram compatibility shim was not applied cleanly" >&2; exit 1; }
python3 -c "import ast, sys, pathlib; ast.parse(pathlib.Path(sys.argv[1]).read_text())" "${dest}/engram.py" \
  || { echo "ERROR the staged Engram module is not valid Python" >&2; exit 1; }

install -o root -g root -m 0644 "${dest}"/*.py
for entry in "${FILES[@]}"; do
  name="${entry%%:*}"
  echo "staged ${dest}/${name} sha256=$(sha256sum "${dest}/${name}" | cut -d' ' -f1)"
done
echo "ENGRAM_PATCHES_OK dest=${dest} revision=${REVISION} files=${#FILES[@]}"
