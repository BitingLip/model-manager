"""
FastAPI routes for model management system
Separated by concern for better organization
"""

from .models import router as models_router
from .workers import router as workers_router
from .health import router as health_router

__all__ = ["health_router", "models_router", "workers_router"]
