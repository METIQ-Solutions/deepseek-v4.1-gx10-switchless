#!/usr/bin/env bash
set -euo pipefail

readonly NCCL_COMMIT="73cf112295c33aee2b895f329f592f2a9b4b0f97"
readonly TAG="deepseek-v4.1-gx10-switchless-switched:local"

case "$(uname -m)" in
  aarch64) ;;
  *) echo "ERROR switched one-HCA build requires linux/arm64 (or explicit arm64 emulation)" >&2; exit 1 ;;
esac
command -v docker >/dev/null 2>&1 || { echo "ERROR docker is required" >&2; exit 1; }
docker build --file docker/Dockerfile.switched-one-hca \
  --build-arg "NCCL_COMMIT=${NCCL_COMMIT}" \
  --tag "${TAG}" .

docker run --rm --pull=never --entrypoint python3 "${TAG}" -c '
import ctypes
lib = ctypes.CDLL("/opt/switched-nccl/libnccl.so.2.30.7")
version = ctypes.c_int()
lib.ncclGetVersion(ctypes.byref(version))
assert version.value == 23007, version.value
print("SWITCHED_NCCL_SOURCE_PASS nccl=23007")
'
docker run --rm --pull=never --entrypoint sha256sum "${TAG}" /opt/switched-nccl/libnccl.so.2.30.7
printf '%s\n' "SWITCHED_NCCL_BUILD_PASS image=${TAG} source_commit=${NCCL_COMMIT}"