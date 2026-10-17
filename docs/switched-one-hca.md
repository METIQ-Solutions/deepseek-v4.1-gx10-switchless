# Switched one-HCA logical-ring adaptation (unvalidated)

This is an isolated Option B adaptation of the historical switchless
implementation at `854204db746a6946c30eb728232e2856473b6fd3`. It is not a
replacement for the original switchless profile and is not deployed by this
repository.

## Required deviation

The current fabric supplies one selected HCA per node and one `/24` address per
rank. The original implementation requires two HCAs, four `/30` lanes and two
listener GIDs. The switched mode therefore makes the smallest explicit
transport relaxation: it keeps Ring-only and Tree/PAT suppression, but permits
exactly one listener GID only when `NCCL_SWITCHED_ONE_HCA=1` is set. The original
switchless mode remains strict.

The current switched logical rank order is `0 -> 1 -> 2 -> 3 -> 0`, and the
transport still uses `NCCL_ALGO=Ring`, native `NET/IB`, `NCCL_CROSS_NIC=0`,
`NCCL_IB_SUBNET_PREFIX_LEN=24`, and the exact HCA/GID selected in
`config/topology.switched-ring-one-hca.env`.

## Artifact plan

`docker/Dockerfile.switched-one-hca` and `docker/build-switched-one-hca.sh`
will build NCCL 2.30.7 from NVIDIA commit
`73cf112295c33aee2b895f329f592f2a9b4b0f97`, apply the historical Ring-only
patches plus the one-HCA switched patch, install it only at
`/opt/switched-nccl/libnccl.so.2.30.7`, and verify the marker, `ncclGetVersion()`
and SHA-256. The artifact SHA-256 is intentionally not claimed until an
ARM64 build is performed; no build is authorized in this task.

## Validation sequence requiring later authorization

1. Run `scripts/validate-switched-topology.py` and the original static tests.
2. Build and verify the source-locked ARM64 library in an isolated builder.
3. Distribute/import it on all four nodes without using the current promoted
   image.
4. Run `scripts/preflight-switched-ring-one-hca.sh` read-only.
5. Run the model-free four-rank collective with loaded-library path/hash,
   `NCCL_ALGO=Ring`, `NET/IB`, HCA/GID and no Socket/Mesh fallback evidence.
6. Only after that evidence is reviewed could a separate full-model start be
   considered.

No remote test, model start, image build or host mutation was performed for
this adaptation.