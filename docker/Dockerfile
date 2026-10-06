# DeepSeek V4.1 Flash on four GB10 / GX10 nodes — switchless TP4 runtime
#
# This image adds exactly one thing to the pinned public four-ring runtime: the
# verified ring-only ("switchless") NCCL 2.30.7 build. Everything else — vLLM,
# PyTorch, CUDA, FlashInfer, the DeepSeek V4.1 model implementation and DSpark
# speculative decoding — comes unmodified from the pinned base image.
#
# Why the replacement is necessary: the stock NCCL still performs Tree/PAT
# transport setup, and on a switchless ring those paths try to reach
# non-neighbour ranks. The ring-only build suppresses them and advertises every
# eligible listener GID from both ConnectX ports of the node.
#
# Build with docker/build.sh, which downloads and checksum-verifies the release
# archive into the build context first.

ARG BASE_IMAGE=ghcr.io/yunwei37/dgx-spark-4-ring-no-switch@sha256:2f8e2a70e73541eacf8c71a8990e907fa18fcd8b2a8d02cd0d54a7c81382a13c

ARG SWITCHLESS_NCCL_ARCHIVE_SHA256=b4a686382a92e57b485ca1bf7cd0f9fde780a68f01ea902ac432b60505b2041f
ARG SWITCHLESS_NCCL_LIBRARY_SHA256=78cb83871792ec57d763d142e4cae26fc754ae284bcc81dcb2a7d50e17d4fa57

FROM ${BASE_IMAGE}
ARG SWITCHLESS_NCCL_ARCHIVE_SHA256
ARG SWITCHLESS_NCCL_LIBRARY_SHA256

USER root

COPY nccl-switchless-release.tar.gz /tmp/nccl-switchless-release.tar.gz

RUN set -eux; \
    printf '%s  %s\n' "${SWITCHLESS_NCCL_ARCHIVE_SHA256}" \
      /tmp/nccl-switchless-release.tar.gz | sha256sum -c -; \
    mkdir -p /tmp/nccl-switchless; \
    tar -xzf /tmp/nccl-switchless-release.tar.gz -C /tmp/nccl-switchless; \
    release_root=/tmp/nccl-switchless/nccl-2.30.7-switchless-hardened-sm121-linux-arm64; \
    test -s "${release_root}/libnccl.so.2.30.7"; \
    mkdir -p /opt/switchless-nccl; \
    cp -a "${release_root}/libnccl.so.2.30.7" /opt/switchless-nccl/; \
    ln -s libnccl.so.2.30.7 /opt/switchless-nccl/libnccl.so.2; \
    ln -s libnccl.so.2 /opt/switchless-nccl/libnccl.so; \
    printf '%s  %s\n' "${SWITCHLESS_NCCL_LIBRARY_SHA256}" \
      /opt/switchless-nccl/libnccl.so.2.30.7 | sha256sum -c -; \
    pip_nccl=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib; \
    test -d "${pip_nccl}"; \
    find "${pip_nccl}" -maxdepth 1 -type f -name 'libnccl.so*' -delete; \
    find "${pip_nccl}" -maxdepth 1 -type l -name 'libnccl.so*' -delete; \
    ln -s /opt/switchless-nccl/libnccl.so.2.30.7 "${pip_nccl}/libnccl.so.2.30.7"; \
    ln -s libnccl.so.2.30.7 "${pip_nccl}/libnccl.so.2"; \
    ln -s libnccl.so.2 "${pip_nccl}/libnccl.so"; \
    test "$(readlink -f "${pip_nccl}/libnccl.so.2")" = /opt/switchless-nccl/libnccl.so.2.30.7; \
    strings /opt/switchless-nccl/libnccl.so.2.30.7 | grep -F 'SWITCHLESS/HARDENED'; \
    sha256sum /opt/switchless-nccl/libnccl.so.2.30.7; \
    rm -rf /tmp/nccl-switchless /tmp/nccl-switchless-release.tar.gz

# PyTorch, DeepEP and vLLM each resolve NCCL through a different path. All of
# them must end up on the same ring-only library, which is why the pip copy is
# replaced by a symlink rather than left beside the new one.
LABEL org.opencontainers.image.title="DeepSeek V4.1 Flash switchless GB10 runtime" \
      org.opencontainers.image.description="Pinned four-ring GB10 runtime with the verified ring-only NCCL 2.30.7" \
      org.opencontainers.image.base.digest="sha256:2f8e2a70e73541eacf8c71a8990e907fa18fcd8b2a8d02cd0d54a7c81382a13c" \
      org.opencontainers.image.licenses="Apache-2.0"
