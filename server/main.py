"""GSStudio FastAPI application entry point."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .auth import BasicAuthMiddleware
from .config import Config
from .db import init_db
from .models import HealthResponse
from .pipeline.worker import PipelineWorker
from .routes import artifacts as artifacts_route
from .routes import tasks as tasks_route
from .static import SPAStaticFiles
from .runtime import get_system_status, probe_gpus

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
    service_token=config.SERVICE_TOKEN,
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
    observed = await asyncio.to_thread(probe_gpus)
    return {**observed, "active_task": worker._current_task_id,
            "mode": "demo" if config.MOCK_PIPELINE else "real"}


@app.get("/api/system/status")
async def system_status():
    return await asyncio.to_thread(get_system_status, config, worker._current_task_id)


# Serve frontend static files in production
web_dist = Path(__file__).parent.parent / "web" / "dist"
if web_dist.exists():
    app.mount("/", SPAStaticFiles(directory=str(web_dist), html=True), name="static")
    logger.info("Serving frontend from %s", web_dist)
