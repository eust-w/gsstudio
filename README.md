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

### Smoke Test

    ./scripts/smoke_test.sh

## Architecture

See [docs/design.md](docs/design.md) for the full architecture document.

## API Reference

See [docs/api.md](docs/api.md) for the REST API reference.

## License

This project is for internal POC use. HY-World 2.0 models are subject to the Tencent Hunyuan Community License.
