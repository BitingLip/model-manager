"""
Simplified FastAPI application for Model Manager
"""

import sys
import os
from pathlib import Path
import logging

# Add the current directory to the Python path
sys.path.append(os.path.abspath("."))

from fastapi import FastAPI, Depends
import uvicorn

from app.models.registry import ModelRegistry
from app.models.downloader import ModelDownloader
from app.services.registry_service import RegistryService
from app.services.download_service import DownloadService
from app.services.model_service import ModelService
from app.config import get_settings
from app.schemas.models import ModelEntry, WorkerInfo, ModelStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Model Manager API",
    description="API for managing ML models",
    version="0.1.0"
)

# Initialize components
settings = get_settings()
registry = ModelRegistry()
downloader = ModelDownloader(registry=registry)
registry_service = RegistryService(registry)
download_service = DownloadService(downloader)
model_service = ModelService(registry_service, download_service)

# Add components to app state
app.state.model_service = model_service
app.state.registry_service = registry_service
app.state.download_service = download_service

# Include API routes
from app.routes import models as models_routes
from app.routes import workers as workers_routes
from app.routes import health as health_routes

app.include_router(models_routes.router)
app.include_router(workers_routes.router)
app.include_router(health_routes.router)

# Dependency for model service
def get_model_service() -> ModelService:
    return app.state.model_service

# Routes
@app.get("/")
async def root():
    return {"message": "Model Manager API is running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "0.1.0"
    }

@app.get("/models")
async def list_models(service: ModelService = Depends(get_model_service)):
    models, count = await service.list_models()
    return {
        "models": [model.model_dump() for model in models],
        "count": count
    }

if __name__ == "__main__":
    # Run without reload for simplicity
    uvicorn.run(app, host="0.0.0.0", port=8085)
