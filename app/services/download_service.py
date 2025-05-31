"""
Simple Download Service Adapter
Provides async interface to the existing ModelDownloader
"""

from typing import Optional, Dict, Any, List
import asyncio
import logging
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
        def _download():
            # Create a basic model entry for the download
            model_entry = ModelEntry(
                id=request.model_id or request.model_name.replace("/", "_"),
                name=request.model_name,
                type=request.model_type,
                size_gb=0.0,
                status=ModelStatus.DOWNLOADING,
                assigned_worker=None,
                download_progress=0.0,
                description="",
                tags=request.tags or [],
                capabilities=[],
                requirements={},
                last_used=None,
                avg_inference_time=None,
                usage_count=0
            )
            
            # Start the download
            success = self.downloader.download_model(
                model_name=request.model_name,
                model_id=model_entry.id
            )
            
            if success:
                model_entry.status = ModelStatus.AVAILABLE
                model_entry.download_progress = 1.0
            else:
                model_entry.status = ModelStatus.ERROR
            
            return model_entry
        
        return await asyncio.to_thread(_download)
    
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
        def _search():
            # Simplified for now - return empty list
            return []
        return await asyncio.to_thread(_search)
    
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
            logger.error("Failed to cleanup DownloadService")
