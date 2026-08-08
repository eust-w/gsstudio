# GSStudio

Full-stack web platform for [HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) - create, monitor, and preview 3D world generation and reconstruction tasks on 5x A800 GPUs.

## Features

- **World Generation**: Text or single image -> panorama -> navigable 3DGS world
- **World Reconstruction**: Multi-view images / video -> 3DGS, point cloud, depth, normals
- **Web Dashboard**: Create tasks, track progress, preview panoramas and 3DGS in browser
- **Task Queue**: Serial execution with checkpoint resume, GPU-aware scheduling
- **3DGS Viewer**: Interactive browser-based Gaussian Splatting viewer

## Quick Start

### Local Development (no GPU)

Run the development script:

    ./scripts/dev_local.sh

This starts the API server (with mock pipeline) on port 8000 and the Vite dev server on port 5173.

### Production Deployment (POC cluster)

Build the image:

    docker build -f Dockerfile.a800 -t gsstudio:latest .

Pre-download model weights:

    ./deploy/scripts/prefetch_models.sh

Deploy to K8s:

    ./deploy/scripts/deploy.sh

### Public POC

The direct public-IP deployment is available at:

    http://120.48.33.45

The direct-IP endpoint is backed by a CCE load balancer and protected with HTTP
Basic Auth. It is plain HTTP. For encrypted access on an untrusted network, use:

    https://geneva-ingredients-expression-located.trycloudflare.com

The TLS endpoint uses a Cloudflare Quick Tunnel. Its URL remains valid while
the `gsstudio-cloudflared` pod is running; recreating that pod generates a new
URL.

The username is `gsstudio`. The generated password is stored in macOS
Keychain under service `gsstudio-poc-basic-auth` and is also injected into the
cluster through the `gsstudio-basic-auth` Kubernetes Secret. It is never stored
in Git.

Full panorama/text world generation also uses the gated `facebook/sam3` model.
Accept its Hugging Face access terms, then provide the approved token through
the optional `gsstudio-huggingface` Secret (`token` key). Multi-view
reconstruction does not require this gated token.

### Smoke Test

    ./scripts/smoke_test.sh

## Architecture

See [docs/design.md](docs/design.md) for the full architecture document.

## API Reference

See [docs/api.md](docs/api.md) for the REST API reference.

## License

This project is for internal POC use. HY-World 2.0 models are subject to the Tencent Hunyuan Community License.
