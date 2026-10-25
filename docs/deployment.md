# Deployment

GSStudio retains the original GPU deployment templates, build helpers, local
mock development script, and historical acceptance reports. These files record
the environments they were written for; they are not a currently hosted service
or a promise that the referenced registry, cluster, or endpoints are available.
Read [SECURITY.md](../SECURITY.md) before running a network-accessible instance.

## Local development

The original `scripts/dev_local.sh` starts a mock API and Vite. Both bind to
`0.0.0.0` (ports 8000 and 5173), so other machines may be able to reach them.
The script installs a small unpinned dependency subset and does not explicitly
disable vLLM management. For an isolated local test, install the full development
requirements and start each process on loopback instead:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
npm --prefix web ci
GSSTUDIO_MOCK_PIPELINE=true GSSTUDIO_MANAGE_VLLM=false \
  .venv/bin/python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
# In another terminal:
npm --prefix web run dev -- --host 127.0.0.1 --port 5173
```

Mock mode simulates task stages. Its PLY/SPZ artifacts are placeholders and do
not demonstrate real scene rendering. World-generation task creation retains
the SAM3 access/cache requirement even in mock mode. A reconstruction task with
two synthetic images exercises the queue without model downloads.

## GPU application image

`Dockerfile.a800` retains the original registry base image and runtime layout.
It is an application overlay: the inherited `a800-v22` image supplies the model
stack, CUDA extensions, API/runtime tools, and upstream checkouts. This
repository does not contain that image or a complete recipe to reproduce it.
Registry credentials must be provided outside the source tree. If the recorded
image is unavailable, prepare a compatible base and edit the `FROM` reference.
There is no `HYWORLD_BASE_IMAGE` build argument in this Dockerfile.

The base is expected to provide:

| Component | Runtime contract |
| --- | --- |
| HY-World | `/opt/hyworld2`, including world generation, reconstruction, and the compatible Gaussian Splatting trainer |
| Panorama Python | `/opt/panogen/bin/python` with HunyuanImage-3 and the expected image-optional interface |
| vLLM Python | `/opt/vllm/bin/python` and the configured Qwen3-VL model integration |
| GPU stack | NVIDIA runtime, compatible CUDA/PyTorch, `torchrun`, and upstream compiled extensions |
| Process control | `supervisord`, `supervisorctl`, Bash, and the API Python dependencies |
| Model storage | Compatible model snapshots, cache layout, and required Torch checkpoints |

The overlay updates selected generic dependencies, installs SpatialIndex and
`rtree`, reapplies four integration patches, and copies the server and built
frontend. License and attribution files are copied into both image variants.
Review [third-party notices](../THIRD_PARTY_NOTICES.md) before obtaining or
redistributing external code, images, or weights.

```sh
npm --prefix web ci
npm --prefix web run build
docker build -f Dockerfile.a800 -t gsstudio:latest .
```

A local `gsstudio:latest` build does not update the immutable image tag in the
recorded GPU deployment. Publish your image to a registry the target cluster
can access and explicitly update the chosen manifest.

`Dockerfile.lite` retains the original Python-only mock image recipe. It is not
the GPU runtime; its dependency installation differs from the complete
`requirements-runtime.txt` reference. The original recipe is preserved and
has not been rebuilt or accepted as part of this source consolidation.

## Kubernetes deployment paths

The original deployment paths have different assumptions:

| Path | Behavior and prerequisites |
| --- | --- |
| `deploy/scripts/deploy.sh` | Creates the namespace and PVCs, applies `deployment.yaml` and `service.yaml`, then applies `ingress.yaml` if the file exists. It does not select a mock variant. |
| `deploy/k8s/deployment.yaml` | Generic PVC-based real GPU deployment (`GSSTUDIO_MOCK_PIPELINE=false`), requesting five GPUs. It does not inject Basic Auth credentials by default. |
| `deploy/k8s/deployment-gpu.yaml` | Recorded five-GPU POC with a fixed node selector, registry image tag, node-local data/model `hostPath` volumes, and registry/Basic Auth/Hugging Face Secret references. |
| `deploy/k8s/deployment-lite.yaml` | Separate original lightweight deployment template. |
| `deploy/k8s/service.yaml` | ClusterIP service exposes the API on port 80 and the internal vLLM service on port 8100. Restrict access to both. |
| `deploy/k8s/service-public-ip.yaml` | Recorded provider-specific load-balancer configuration; review before applying to any cluster. |
| `deploy/k8s/cloudflared-quick-tunnel.yaml` | Temporary tunnel helper; recreating its pod changes the generated URL. |
| `deploy/k8s/build-context-loader.yaml`, `kaniko-build.yaml` | Original in-cluster build helpers with their original storage and registry assumptions. |
| `deploy/k8s/preflight-v50.yaml` | Historical release preflight, not a general live-health certificate. |

Before applying any file, inspect the current Kubernetes context and replace
node, registry, storage, ingress, and endpoint configuration for your own
infrastructure. The generic PVC manifest requests 250 GiB for model cache and
500 GiB for task storage. The recorded POC mounts local paths instead. Do not
mix the two storage approaches without adapting the manifests.

Keep one replica and one API process because SQLite and an in-process worker
coordinate the task queue. Review CPU, memory, shared-memory, GPU, and disk
requirements against the actual model workload.

## Models and credentials

The standalone and Kubernetes branches of `deploy/scripts/prefetch_models.sh`
retain their original, different download lists. The standalone branch fetches
the broad model set. The Kubernetes branch fills selected caches on the
recorded node using its recorded image and `hostPath`. Review both branches and
complete the cache layout for your runtime before starting inference.

World generation uses gated `facebook/sam3`; accept its applicable access terms
and provide an approved `HF_TOKEN` when downloading. A complete offline cache
can avoid retaining the token. Multi-view reconstruction does not require the
SAM3 token. Other upstream licenses and model dependencies still apply.

The GPU manifest reads credentials from existing Kubernetes Secrets:

- `gsstudio-basic-auth`: `username` and `password` keys.
- `gsstudio-huggingface`: optional `token` key.
- `gsstudio-registry`: registry pull credentials.

These are resource names, not credential values. Provision secrets with your
secret manager, keep each password independent of the username, and never
commit values in manifests, `.env` files, logs, or build contexts. Basic Auth
can also be configured with `GSSTUDIO_BASIC_AUTH_USER` and
`GSSTUDIO_BASIC_AUTH_PASSWORD`; it is disabled unless both are set.

## Validation

`GET /api/health` checks API liveness only. Mock GPU status is synthetic. In
real mode, `GET /api/gpu/status` queries `nvidia-smi` but can return an empty
list on failure. Neither endpoint proves model execution works.

For real acceptance, inspect GPU and supervised-process state, run a task with
inputs you are allowed to use, wait for its final state, inspect artifacts and
quality reports, and verify panorama/3DGS rendering in a browser. For real
world generation, confirm `artifact_validation.valid` alongside completion.
Historical reports in `docs/experiments/` describe past observations and do not
establish present service availability. See [release validation](release-validation.md)
for the current source-check scope.
