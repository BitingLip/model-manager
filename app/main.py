"""
FastAPI application entry point for Model Manager
Implements proper lifespan management and middleware
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import structlog
import uvicorn

from app.config import get_settings
from app.core.logging_config import setup_logging
from app.core.dependencies import get_model_service
from app.routes import health_router, models_router, workers_router

# Initialize logging
settings = get_settings()
setup_logging(settings)
logger = structlog.get_logger(__name__)

# Global model service instance
model_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global model_service
    
    # Startup
    logger.info("Starting Model Manager application")
    
    try:
        settings = get_settings()
          # Initialize core components
        from app.models.postgresql_registry import PostgreSQLModelRegistry
        from app.models.downloader import ModelDownloader
        from app.services import RegistryService, DownloadService, ModelService
        
        # Database configuration for PostgreSQL
        db_config = {
            'host': os.getenv('MODEL_DB_HOST', 'localhost'),
            'port': int(os.getenv('MODEL_DB_PORT', '5432')),
            'database': os.getenv('MODEL_DB_NAME', 'bitinglip_models'),
            'user': os.getenv('MODEL_DB_USER', 'model_manager'),
            'password': os.getenv('MODEL_DB_PASSWORD', 'model_manager_2025!'),
        }
        
        # Create instances - using PostgreSQL registry
        registry = PostgreSQLModelRegistry(db_config)
        models_dir = Path(os.getenv('MODELS_DIRECTORY', './models'))
        models_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
          # Create adapter services with full HuggingFace integration
        registry_service = RegistryService(registry)
        
        # Initialize ModelDownloader with HuggingFace Hub integration
        downloader = ModelDownloader(
            registry=registry,
            download_dir=models_dir,
            cache_dir=Path('./cache/downloads'),
            max_concurrent_downloads=3
        )
        
        # Create DownloadService adapter
        download_service = DownloadService(downloader)
        
        # Create full ModelService with download capabilities
        model_service = ModelService(registry_service, download_service)
        
        # Store in app state for dependency injection
        app.state.registry_service = registry_service
        app.state.model_service = model_service
        
        # Store global reference for cleanup
        model_service = model_service
        
        logger.info("Model Manager startup completed", 
                   database=db_config['database'],
                   models_dir=str(models_dir))
        
        yield
        
    except Exception as e:
        logger.error("Failed to start Model Manager", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Model Manager application")
        model_service = None


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title="Model Manager",
        description="AI Model Management System with HuggingFace Integration",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
      # Add routers
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(workers_router)
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler for unhandled errors"""
        logger.error("Unhandled exception", 
                    method=request.method,
                    url=str(request.url),
                    error=str(exc),
                    exc_info=exc)
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Internal server error",
                "error": "An unexpected error occurred"
            }
        )
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "service": "Model Manager",
            "version": "1.0.0",
            "description": "AI Model Management System",
            "endpoints": {
                "health": "/health",
                "models": "/models",
                "workers": "/workers",
                "docs": "/docs"
            }
        }
    
    # API versioning endpoint
    @app.get("/v1")
    async def api_v1():
        """API v1 endpoint"""
        return {
            "version": "1.0.0",
            "endpoints": {
                "models": "/models",
                "workers": "/workers",
                "health": "/health"
            }
        }
    
    return app


# Create app instance
app = create_app()


def main():
    """Main entry point for running the application"""
    settings = get_settings()
    
    # Use INFO as default log level since settings doesn't have log_level
    log_level = getattr(settings, 'log_level', 'info')
    if hasattr(log_level, 'lower'):
        log_level = log_level.lower()
    else:
        log_level = 'info'
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()
