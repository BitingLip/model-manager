"""
FastAPI application entry point for Model Manager
Implements proper lifespan management and middleware
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import structlog
import uvicorn

from .config import get_settings
from .core.logging_config import setup_logging
from .core.dependencies import get_model_service
from .routes import models_router, workers_router, health_router

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
        settings = get_settings()        # Initialize core components
        from .models.registry import ModelRegistry
        from .models.downloader import ModelDownloader
        from .services.registry_service import RegistryService
        from .services.download_service import DownloadService
        from .services.model_service import ModelService
        
        # Create instances
        registry = ModelRegistry(db_path=Path(settings.registry_file))
        downloader = ModelDownloader(registry=registry, download_dir=Path(settings.models_directory))
        
        # Create adapter services
        registry_service = RegistryService(registry)
        download_service = DownloadService(downloader)
        model_service = ModelService(registry_service, download_service)
        
        # Store in app state for dependency injection
        app.state.registry_service = registry_service
        app.state.download_service = download_service
        app.state.model_service = model_service
        
        logger.info("Model Manager startup completed", 
                   db_path=str(settings.registry_file),
                   models_dir=str(settings.models_directory))
        
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
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )


if __name__ == "__main__":
    main()
