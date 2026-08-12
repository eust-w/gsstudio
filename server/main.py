"""GSStudio FastAPI application entry point."""
from __future__ import annotations

import asyncio
import logging
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .auth import BasicAuthMiddleware
from .config import Config
from .db import init_db, list_tasks
from .models import GPUInfo, GPUStatus, HealthResponse
from .pipeline.worker import PipelineWorker
from .routes import artifacts as artifacts_route
from .routes import tasks as tasks_route
from .static import SPAStaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

config = Config()
worker = PipelineWorker(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.tasks_dir.mkdir(parents=True, exist_ok=True)
    init_db(config.DB_PATH)
    removed = tasks_route.cleanup_stale_incoming(config)
    if removed:
        logger.info("Removed %d abandoned incoming task directories", removed)
    logger.info("Database initialized at %s", config.DB_PATH)

    # Initialize routers
    tasks_route.init(config, worker)
    artifacts_route.init(config)

    # Start pipeline worker in background
    worker_task = asyncio.create_task(worker.start())
    logger.info("Pipeline worker started")

    yield

    # Shutdown
    await worker.stop()
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutdown complete")


app = FastAPI(
    title="GSStudio",
    description="HY-World 2.0 Full-Stack Web Platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    BasicAuthMiddleware,
    username=config.BASIC_AUTH_USER,
    password=config.BASIC_AUTH_PASSWORD,
    exempt_paths=("/api/health",),
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(tasks_route.router, prefix="/api")
app.include_router(artifacts_route.router, prefix="/api")


@app.get("/api/health")
async def health():
    return HealthResponse()


@app.get("/api/gpu/status")
async def gpu_status():
    if config.MOCK_PIPELINE:
        # Return mock GPU status for local dev
        return GPUStatus(
            gpus=[
                GPUInfo(index=i, name="A800-SXM4-80GB", used_by="vllm" if i == 0 else None,
                        vram_used_gb=48.0 if i == 0 else 0.0)
                for i in range(config.NUM_GPUS)
            ],
            active_task=worker._current_task_id,
        )

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append(GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    vram_total_gb=float(parts[2]) / 1024,
                    vram_used_gb=float(parts[3]) / 1024,
                    used_by="vllm" if int(parts[0]) == 0 else None,
                ))
        return GPUStatus(gpus=gpus, active_task=worker._current_task_id)
    except Exception as e:
        return GPUStatus(gpus=[], active_task=None)


# Serve frontend static files in production
web_dist = Path(__file__).parent.parent / "web" / "dist"
if web_dist.exists():
    app.mount("/", SPAStaticFiles(directory=str(web_dist), html=True), name="static")
    logger.info("Serving frontend from %s", web_dist)
