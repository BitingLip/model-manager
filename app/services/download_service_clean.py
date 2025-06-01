"""
Simple Download Service Adapter
Provides async interface to the existing ModelDownloader
"""

from typing import Optional, Dict, Any, List
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from ..schemas.models import ModelEntry, ModelDownloadRequest, ModelStatus
from ..models.downloader import ModelDownloader

logger = logging.getLogger(__name__)


class DownloadService:
    """Adapter service for ModelDownloader with async interface"""
    
    def __init__(self, downloader: ModelDownloader):
        self.downloader = downloader
        logger.info("DownloadService adapter initialized")
    
    async def download_model(self, request: ModelDownloadRequest) -> ModelEntry:
        """Download a model from HuggingFace"""
        # Start the download using the async method
        success = await self.downloader.download_model(
            model_name=request.model_name,
            model_id=request.model_id or request.model_name.replace("/", "_")
        )
        
        if success:
            # Get the registered model from the downloader's registry
            model_id = request.model_id or request.model_name.replace("/", "_")
            model_entry = self.downloader.registry.get_model(model_id)
            if model_entry:
                return model_entry
        
        # If download failed or model not found, return a basic entry with error status
        return ModelEntry(
            id=request.model_id or request.model_name.replace("/", "_"),
            name=request.model_name,
            type=request.model_type,
            size_gb=0.0,
            status=ModelStatus.ERROR,
            assigned_worker=None,
            download_progress=0.0,
            description="Download failed",
            tags=[],
            capabilities=[],
            requirements={},            created_at=datetime.now(),
            updated_at=datetime.now(),
            last_used=None,
            avg_inference_time=None,
            usage_count=0
        )
    
    async def get_download_progress(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get download progress"""
        def _get_progress():
            return self.downloader.get_download_progress(model_id)
        return await asyncio.to_thread(_get_progress)
    
    async def cancel_download(self, model_id: str) -> bool:
        """Cancel active download"""
        def _cancel():
            return self.downloader.cancel_download(model_id)
        return await asyncio.to_thread(_cancel)
    
    async def delete_model_files(self, model_id: str) -> bool:
        """Delete model files from disk"""
        def _delete():
            return self.downloader.delete_model(model_id)
        return await asyncio.to_thread(_delete)
    
    async def search_huggingface_models(self, 
                                      query: str,
                                      model_type: Optional[str] = None,
                                      limit: int = 20) -> List[Dict[str, Any]]:
        """Search HuggingFace models"""
        return await self.downloader.search_huggingface_models(query, model_type, "downloads", limit)
    
    async def list_downloads(self) -> Dict[str, Dict[str, Any]]:
        """List active downloads"""
        def _list():
            return self.downloader.list_active_downloads()
        return await asyncio.to_thread(_list)
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.downloader.cleanup()
            logger.info("DownloadService cleanup completed")
        except Exception as e:
            logger.error(f"Failed to cleanup DownloadService: {str(e)}")
