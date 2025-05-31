"""
Model Management System for AMD GPU Cluster
Provides unified model registry, download management, and worker assignment
"""

from .registry import ModelRegistry
from .schemas import ModelEntry, WorkerInfo, ModelType, ModelStatus, WorkerStatus
from .downloader import ModelDownloader

__version__ = "1.0.0"
__all__ = [
    "ModelRegistry",
    "ModelEntry", 
    "WorkerInfo",
    "ModelType",
    "ModelStatus",
    "WorkerStatus",
    "ModelDownloader"
]
