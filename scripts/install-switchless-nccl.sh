#!/usr/bin/env bash
#
# Install the verified ring-only NCCL 2.30.7 build on THIS host.
#
# Runs on the target node (it is also piped over SSH by scripts/preflight.sh and
# scripts/start.sh, which is why it depends on nothing but curl, tar and
# sha256sum). Idempotent: an already-correct /opt/switchless-nccl is left alone.
#
# The archive is never redistributed by this repository; it is fetched from the
# pinned upstream release and verified twice — once as the archive, once as the
# installed library — because the library is what the container actually loads.
#
# Usage: sudo bash install-switchless-nccl.sh [--dest /opt/switchless-nccl]

set -euo pipefail

readonly RELEASE_URL="https://github.com/alexellis/switchless-nccl/releases/download/v0.0.1/nccl-2.30.7-switchless-hardened-sm121-linux-arm64.tar.gz"
readonly ARCHIVE_SHA256="b4a686382a92e57b485ca1bf7cd0f9fde780a68f01ea902ac432b60505b2041f"
readonly LIBRARY_SHA256="78cb83871792ec57d763d142e4cae26fc754ae284bcc81dcb2a7d50e17d4fa57"
readonly RELEASE_ROOT="nccl-2.30.7-switchless-hardened-sm121-linux-arm64"

dest=/opt/switchless-nccl
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dest) [ "$#" -ge 2 ] || { echo "ERROR --dest requires a value" >&2; exit 2; }; dest="$2"; shift 2 ;;
    -h|--help) sed -n '3,14p' "$0"; exit 0 ;;
    *) echo "ERROR unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ "$(id -u)" -ne 0 ]; then
  echo "ERROR must run as root (the library is mounted read-only into the container)" >&2
  exit 1
fi

library="${dest}/libnccl.so.2.30.7"
if [ -f "${library}" ] && [ "$(sha256sum "${library}" | cut -d' ' -f1)" = "${LIBRARY_SHA256}" ]; then
  echo "SWITCHLESS_NCCL_OK path=${library} (already installed)"
  exit 0
fi

command -v curl >/dev/null 2>&1 || { echo "ERROR curl is required" >&2; exit 1; }
command -v sha256sum >/dev/null 2>&1 || { echo "ERROR sha256sum is required" >&2; exit 1; }
command -v tar >/dev/null 2>&1 || { echo "ERROR tar is required" >&2; exit 1; }

work_dir="$(mktemp -d --tmpdir switchless-nccl.XXXXXX)"
trap 'rm -rf -- "${work_dir}"' EXIT

echo "==> downloading the pinned ring-only NCCL release"
curl --fail --location --silent --show-error --retry 5 \
  --connect-timeout 15 --max-time 900 \
  --output "${work_dir}/release.tar.gz" "${RELEASE_URL}"

printf '%s  %s\n' "${ARCHIVE_SHA256}" "${work_dir}/release.tar.gz" | sha256sum -c -

tar -xzf "${work_dir}/release.tar.gz" -C "${work_dir}"
source_library="${work_dir}/${RELEASE_ROOT}/libnccl.so.2.30.7"
[ -s "${source_library}" ] || { echo "ERROR archive does not contain ${RELEASE_ROOT}/libnccl.so.2.30.7" >&2; exit 1; }

# The ring-only behaviour is identifiable from the build's own marker; the
# SHA-256 alone would not tell a future reader what was verified.
strings "${source_library}" | grep -F 'SWITCHLESS/HARDENED' >/dev/null \
  || { echo "ERROR the archive is not the hardened switchless build" >&2; exit 1; }
printf '%s  %s\n' "${LIBRARY_SHA256}" "${source_library}" | sha256sum -c -

install -d -o root -g root -m 0755 "${dest}"
install -o root -g root -m 0644 "${source_library}" "${library}"
ln -sfn libnccl.so.2.30.7 "${dest}/libnccl.so.2"
ln -sfn libnccl.so.2 "${dest}/libnccl.so"

printf '%s  %s\n' "${LIBRARY_SHA256}" "${library}" | sha256sum -c -

echo "SWITCHLESS_NCCL_OK path=${library} sha256=${LIBRARY_SHA256}"
