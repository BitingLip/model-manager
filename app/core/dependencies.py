"""
Dependency injection for Model Manager services
"""

from fastapi import Request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..services.model_service import ModelService
    from ..services.download_service import DownloadService
    from ..services.registry_service import RegistryService


def get_model_service(request: Request) -> "ModelService":
    """Dependency to get model service from app state"""
    return request.app.state.model_service


def get_download_service(request: Request) -> "DownloadService":
    """Dependency to get download service from app state"""
    return request.app.state.download_service


def get_registry_service(request: Request) -> "RegistryService":
    """Dependency to get registry service from app state"""
    return request.app.state.registry_service
