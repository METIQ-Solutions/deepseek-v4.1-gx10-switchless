#!/usr/bin/env bash
#
# Build the switchless DeepSeek V4.1 runtime image.
#
# The only external input is the pinned public base image plus the verified
# ring-only NCCL release archive. Nothing is pulled from or pushed to a private
# registry; the resulting image is tagged locally unless you pass a --tag.
#
# Usage:
#   docker/build.sh                       # tag: deepseek-v4.1-gx10-switchless:local
#   docker/build.sh --tag my/name:1
#   docker/build.sh --no-cache
#
# Requirements: docker with a working buildx/BuildKit or classic builder,
# curl, sha256sum, and enough disk for the ~30 GB base image.

set -euo pipefail

readonly BASE_IMAGE="ghcr.io/yunwei37/dgx-spark-4-ring-no-switch@sha256:2f8e2a70e73541eacf8c71a8990e907fa18fcd8b2a8d02cd0d54a7c81382a13c"
readonly NCCL_RELEASE_URL="https://github.com/alexellis/switchless-nccl/releases/download/v0.0.1/nccl-2.30.7-switchless-hardened-sm121-linux-arm64.tar.gz"
readonly NCCL_ARCHIVE_SHA256="b4a686382a92e57b485ca1bf7cd0f9fde780a68f01ea902ac432b60505b2041f"
readonly NCCL_LIBRARY_SHA256="78cb83871792ec57d763d142e4cae26fc754ae284bcc81dcb2a7d50e17d4fa57"
readonly NCCL_RELEASE_ROOT="nccl-2.30.7-switchless-hardened-sm121-linux-arm64"

tag="deepseek-v4.1-gx10-switchless:local"
extra_args=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag)
      [ "$#" -ge 2 ] || { echo "ERROR --tag requires a value" >&2; exit 2; }
      tag="$2"; shift 2 ;;
    --no-cache|--pull|--progress=*)
      extra_args+=("$1"); shift ;;
    -h|--help)
      sed -n '3,15p' "$0"; exit 0 ;;
    *)
      echo "ERROR unknown argument: $1" >&2; exit 2 ;;
  esac
done

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [ "$(uname -m)" != "aarch64" ]; then
  echo "ERROR this image is linux/arm64 only (detected $(uname -m)); run the build on a GB10/GX10 node or under arm64 emulation" >&2
  exit 1
fi

command -v docker >/dev/null 2>&1 || { echo "ERROR docker is required" >&2; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "ERROR curl is required" >&2; exit 1; }

work_dir="$(mktemp -d --tmpdir v4.1-switchless-build.XXXXXX)"
trap 'rm -rf -- "${work_dir}"' EXIT

echo "==> fetching the pinned ring-only NCCL release"
curl --fail --location --silent --show-error --retry 5 \
  --connect-timeout 15 --max-time 900 \
  --output "${work_dir}/nccl-switchless-release.tar.gz" \
  "${NCCL_RELEASE_URL}"

printf '%s  %s\n' "${NCCL_ARCHIVE_SHA256}" "${work_dir}/nccl-switchless-release.tar.gz" | sha256sum -c -

echo "==> verifying the library inside the archive"
tar -xzf "${work_dir}/nccl-switchless-release.tar.gz" -C "${work_dir}"
# The hardened build is identified by its own marker string; a release archive
# that unpacks but does not carry it is not the artifact this runtime qualified.
strings "${work_dir}/${NCCL_RELEASE_ROOT}/libnccl.so.2.30.7" | grep -F 'SWITCHLESS/HARDENED' >/dev/null
printf '%s  %s\n' "${NCCL_LIBRARY_SHA256}" \
  "${work_dir}/${NCCL_RELEASE_ROOT}/libnccl.so.2.30.7" | sha256sum -c -

# The Dockerfile COPYs the archive from the build context, so stage it there
# temporarily and always remove it again.
cp -- "${work_dir}/nccl-switchless-release.tar.gz" "${repo_root}/nccl-switchless-release.tar.gz"
trap 'rm -rf -- "${work_dir}" "${repo_root}/nccl-switchless-release.tar.gz"' EXIT

echo "==> building ${tag}"
docker build \
  --file docker/Dockerfile \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "SWITCHLESS_NCCL_ARCHIVE_SHA256=${NCCL_ARCHIVE_SHA256}" \
  --build-arg "SWITCHLESS_NCCL_LIBRARY_SHA256=${NCCL_LIBRARY_SHA256}" \
  "${extra_args[@]+"${extra_args[@]}"}" \
  --tag "${tag}" \
  .

echo "==> verifying the built image"
docker run --rm --entrypoint /bin/bash "${tag}" -c '
  set -euo pipefail
  test "$(readlink -f /opt/switchless-nccl/libnccl.so.2)" = /opt/switchless-nccl/libnccl.so.2.30.7
  printf "%s  %s\n" '"${NCCL_LIBRARY_SHA256}"' /opt/switchless-nccl/libnccl.so.2.30.7 | sha256sum -c -
  python3 -c "
import ctypes
library = ctypes.CDLL(\"/opt/switchless-nccl/libnccl.so.2.30.7\")
version = ctypes.c_int()
library.ncclGetVersion(ctypes.byref(version))
assert version.value == 23007, version.value
print(\"ncclGetVersion()=23007\")
"
'

echo
echo "BUILD_PASS image=${tag}"
echo "Next: distribute the image to every rank (docker save | ssh … docker load,"
echo "or your own registry), then run scripts/preflight.sh."
