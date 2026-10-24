# GSStudio

Full-stack web platform for [HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) - create, monitor, and preview 3D world generation and reconstruction tasks on 5x A800 GPUs.

## Features

- **World Generation**: Text or single image -> panorama -> navigable 3DGS world
- **World Reconstruction**: Multi-view images / video -> 3DGS, point cloud, depth, normals
- **Web Dashboard**: Create tasks, track progress, preview panoramas and 3DGS in browser
- **Task Queue**: Serial execution with checkpoint resume, GPU-aware scheduling
- **3DGS Viewer**: Interactive browser-based Gaussian Splatting viewer
- **Delivery Quality**: High-quality generation trains `strict_3d` from at least
  six accepted trajectories, `degraded_3d` from one to five plus RGB-D anchors,
  or `anchored_3d` from panorama/polar anchors when none pass. Completion requires
  validated 3DGS artifacts; rejected trajectories are excluded from training.

The last recorded GPU/public acceptance is [the 2026-08-12 `a800-v50`
release](docs/experiments/2026-08-12-guaranteed-3dgs-delivery.md). As of
2026-10-25, current online service and GPU availability remain unverified;
see [the current investigation](docs/experiments/2026-10-25-service-status-and-source-retry.md)
for reachability evidence and validation results.

## Quick Start

### Local Development (no GPU)

Run the development script:

    ./scripts/dev_local.sh

This starts the API server (with mock pipeline) on port 8000 and the Vite dev server on port 5173.

### Production Deployment (POC cluster)

Build the frontend, then the image (the Dockerfile copies `web/dist`):

    npm --prefix web ci
    npm --prefix web run build
    docker build -f Dockerfile.a800 -t gsstudio:latest .

Pre-download model weights:

    ./deploy/scripts/prefetch_models.sh

The recorded A800 POC uses [deployment-gpu.yaml](deploy/k8s/deployment-gpu.yaml),
which pins `a800-v50` to its GPU node and mounts node-local data/model directories.
Inspect the Kubernetes context and existing deployment before applying it:

    kubectl config current-context
    kubectl -n gsstudio get deployment gsstudio
    kubectl apply -f deploy/k8s/deployment-gpu.yaml

Registry credentials, Basic Auth, cached model weights, and the GPU node must
already be provisioned. `deploy/scripts/deploy.sh` instead applies the generic
PVC-based `deployment.yaml`; it does not reproduce the recorded A800 POC. To
deploy a new build, publish it under a new immutable registry tag and update
the GPU manifest's image; building local `gsstudio:latest` alone does not update
the pinned deployment.

### Public POC

The public-IP endpoint recorded during the August acceptance was:

    http://120.48.33.45

The direct-IP deployment uses a CCE load balancer and HTTP Basic Auth over plain
HTTP. Its recorded Cloudflare TLS endpoint was:

    https://geneva-ingredients-expression-located.trycloudflare.com

The TLS endpoint uses a Cloudflare Quick Tunnel. Its URL remains valid while
the `gsstudio-cloudflared` pod is running; recreating that pod generates a new
URL. Both endpoints require a fresh reachability check before use; the URLs
above do not establish current availability.

The username is `<configured-user>`. The password is stored in macOS Keychain under account
`<configured-user>`, service `gsstudio-poc-basic-auth`, and is also injected into the cluster
through the `gsstudio-basic-auth` Kubernetes Secret. It is never stored in Git.

Full panorama/text world generation also uses the gated `facebook/sam3` model.
Accept its Hugging Face access terms, then provide the approved token through
the optional `gsstudio-huggingface` Secret (`token` key). Multi-view
reconstruction does not require this gated token.

### Local Verification

    GSSTUDIO_MANAGE_VLLM=false .venv/bin/python -m pytest -q
    npm --prefix web test
    npm --prefix web run build

These tests use mocks and do not establish real GPU inference health.

### Smoke Test

    ./scripts/smoke_test.sh

This script targets an unauthenticated local API by default, creates a task,
and checks that it is queryable. It neither supplies production Basic Auth nor
waits for valid 3DGS output. Production acceptance additionally needs GPU/process
inspection, a real completed task, artifact validation, and browser preview.

## Architecture

See [docs/design.md](docs/design.md) for the full architecture document.

## API Reference

See [docs/api.md](docs/api.md) for the REST API reference.

## License

This project is for internal POC use. HY-World 2.0 models are subject to the Tencent Hunyuan Community License.
