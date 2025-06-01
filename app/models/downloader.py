"""
Enhanced Model Downloader with HuggingFace Integration
Provides smart downloading, progress tracking, and automatic registration
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Union
from datetime import datetime
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

import structlog
from huggingface_hub import (
    hf_hub_download, 
    snapshot_download,
    HfApi,
    repo_info,
    HfFolder,
    list_models
)
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

from .registry import ModelRegistry
from .postgresql_registry import PostgreSQLModelRegistry
from ..schemas.models import ModelEntry, ModelType, ModelStatus

logger = structlog.get_logger(__name__)


class DownloadProgress:
    """Track download progress for a model"""
    def __init__(self, model_id: str, total_size: int = 0):
        self.model_id = model_id
        self.total_size = total_size
        self.downloaded_size = 0
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        self.download_speed = 0.0  # MB/s
        self.status = "initializing"
        
    def update(self, downloaded: int):
        """Update progress"""
        now = datetime.now()
        time_diff = (now - self.last_update).total_seconds()
        
        if time_diff > 0:
            size_diff = downloaded - self.downloaded_size
            self.download_speed = (size_diff / (1024 * 1024)) / time_diff
            
        self.downloaded_size = downloaded
        self.last_update = now
        self.status = "downloading"
        
    def complete(self):
        """Mark as completed"""
        self.status = "completed"
        self.downloaded_size = self.total_size
        
    def error(self, message: str):
        """Mark as error"""
        self.status = f"error: {message}"
        
    @property
    def progress_percent(self) -> float:
        """Get progress percentage"""
        if self.total_size == 0:
            return 0.0
        return (self.downloaded_size / self.total_size) * 100


class ModelDownloader:
    """Enhanced model downloader with HuggingFace integration"""
    
    def __init__(self, 
                 registry: Union[ModelRegistry, PostgreSQLModelRegistry],
                 download_dir: Optional[Path] = None,
                 cache_dir: Optional[Path] = None,
                 max_concurrent_downloads: int = 3):
        """
        Initialize downloader
        
        Args:
            registry: Model registry instance
            download_dir: Directory to store downloaded models
            cache_dir: Cache directory for temporary files
            max_concurrent_downloads: Maximum concurrent downloads
        """
        self.registry = registry
        self.download_dir = download_dir or Path("./models")
        self.cache_dir = cache_dir or Path("./cache/downloads")
        self.max_concurrent_downloads = max_concurrent_downloads
        
        # Create directories
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize HuggingFace API
        self.hf_api = HfApi()
        
        # Progress tracking
        self.active_downloads: Dict[str, DownloadProgress] = {}
        self.download_callbacks: Dict[str, List[Callable]] = {}
        
        # Thread pool for concurrent downloads
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_downloads)
        
        logger.info("Model downloader initialized", 
                   download_dir=str(self.download_dir),
                   cache_dir=str(self.cache_dir))

    def add_progress_callback(self, model_id: str, callback: Callable[[Dict[str, Any]], None]):
        """Add a progress callback for a model download"""
        if model_id not in self.download_callbacks:
            self.download_callbacks[model_id] = []
        self.download_callbacks[model_id].append(callback)

    def _notify_progress(self, model_id: str, progress_data: Dict[str, Any]):
        """Notify all callbacks about progress update"""
        if model_id in self.download_callbacks:
            for callback in self.download_callbacks[model_id]:
                try:
                    callback(progress_data)
                except Exception as e:
                    logger.warning("Progress callback failed", model_id=model_id, error=str(e))

    def get_model_type_from_config(self, model_info: Dict[str, Any]) -> ModelType:
        """Determine model type from HuggingFace model info"""
        pipeline_tag = model_info.get('pipeline_tag', '')
        tags = model_info.get('tags', [])
        
        # Map pipeline tags to our model types
        if pipeline_tag in ['text-generation', 'text2text-generation']:
            return ModelType.LLM
        elif pipeline_tag in ['image-classification', 'object-detection', 'image-segmentation']:
            return ModelType.VISION
        elif pipeline_tag in ['text-to-speech', 'automatic-speech-recognition']:
            return ModelType.TTS
        elif pipeline_tag in ['text-to-image', 'image-to-image']:
            return ModelType.DIFFUSION
        elif pipeline_tag in ['feature-extraction', 'sentence-similarity']:
            return ModelType.EMBEDDING
        
        # Check tags for additional hints
        if any(tag in tags for tag in ['llm', 'language-model', 'gpt', 'bert']):
            return ModelType.LLM
        elif any(tag in tags for tag in ['computer-vision', 'cv']):
            return ModelType.VISION
        elif any(tag in tags for tag in ['tts', 'speech']):
            return ModelType.TTS
        elif any(tag in tags for tag in ['diffusion', 'stable-diffusion']):
            return ModelType.DIFFUSION
        elif any(tag in tags for tag in ['embedding', 'sentence-transformers']):
            return ModelType.EMBEDDING
        
        return ModelType.OTHER

    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information from HuggingFace"""
        try:
            # Use repo_info to get model information
            info = repo_info(repo_id=model_name, repo_type="model")
            
            # Extract relevant information
            model_info = {
                'name': model_name,
                'author': info.author if hasattr(info, 'author') else model_name.split('/')[0],
                'description': getattr(info, 'description', ''),
                'tags': getattr(info, 'tags', []),
                'pipeline_tag': getattr(info, 'pipeline_tag', ''),
                'library_name': getattr(info, 'library_name', ''),
                'created_at': getattr(info, 'created_at', None),
                'last_modified': getattr(info, 'last_modified', None)
            }
            
            return model_info
            
        except (RepositoryNotFoundError, RevisionNotFoundError) as e:
            logger.error("Model not found", model_name=model_name, error=str(e))
            raise
        except Exception as e:
            logger.error("Failed to get model info", model_name=model_name, error=str(e))
            raise

    def _create_progress_callback(self, model_id: str):
        """Create a progress callback function for HuggingFace downloads"""
        def progress_callback(progress_dict):
            if model_id in self.active_downloads:
                progress = self.active_downloads[model_id]
                # HuggingFace progress format may vary
                if 'downloaded' in progress_dict and 'total' in progress_dict:
                    progress.total_size = progress_dict['total']
                    progress.update(progress_dict['downloaded'])
                
                # Notify callbacks
                self._notify_progress(model_id, {
                    'model_id': model_id,
                    'progress_percent': progress.progress_percent,
                    'downloaded_mb': progress.downloaded_size / (1024 * 1024),
                    'total_mb': progress.total_size / (1024 * 1024),
                    'speed_mbps': progress.download_speed,
                    'status': progress.status
                })
        
        return progress_callback

    async def download_model(self, 
                           model_name: str, 
                           model_id: Optional[str] = None,
                           revision: str = "main",
                           cache_dir: Optional[Path] = None) -> bool:
        """
        Download a model from HuggingFace
        
        Args:
            model_name: HuggingFace model name (e.g., 'microsoft/DialoGPT-small')
            model_id: Custom model ID for registry (defaults to model_name)
            revision: Model revision/branch
            cache_dir: Custom cache directory
            
        Returns:
            True if download successful, False otherwise
        """
        if not model_id:
            model_id = model_name.replace('/', '_')
            
        logger.info("Starting model download", model_name=model_name, model_id=model_id)
        
        # Check if already downloading
        if model_id in self.active_downloads:
            logger.warning("Model already downloading", model_id=model_id)
            return False
        
        try:
            # Get model info first
            model_info = await self.get_model_info(model_name)
            
            # Check if model already exists
            existing_model = self.registry.get_model(model_id)
            if existing_model and existing_model.status == ModelStatus.AVAILABLE:
                logger.info("Model already available", model_id=model_id)
                return True
            
            # Initialize progress tracking
            progress = DownloadProgress(model_id)
            self.active_downloads[model_id] = progress
            
            # Determine model type
            model_type = self.get_model_type_from_config(model_info)
            
            # Create or update model entry
            if not existing_model:
                model_entry = ModelEntry(
                    id=model_id,
                    name=model_name,
                    type=model_type,
                    status=ModelStatus.DOWNLOADING,
                    size_gb=0.0,  # Will be updated after download
                    assigned_worker=None,
                    download_progress=0.0,
                    last_used=None,
                    avg_inference_time=0.0,
                    usage_count=0,
                    description=model_info.get('description', ''),
                    tags=model_info.get('tags', []),
                    capabilities=[model_info.get('pipeline_tag', '')] if model_info.get('pipeline_tag') else [],
                    requirements={
                        'library': model_info.get('library_name', ''),
                        'hf_model_name': model_name
                    }
                )
                
                if not self.registry.register_model(model_entry):
                    logger.error("Failed to register model", model_id=model_id)
                    return False
            
            # Update status to downloading
            self.registry.update_model_status(model_id, ModelStatus.DOWNLOADING)
            
            # Download using HuggingFace hub
            download_path = self.download_dir / model_id
            cache_path = cache_dir or self.cache_dir
            
            # Create progress callback
            progress_callback = self._create_progress_callback(model_id)
            
            # Download model using snapshot_download
            local_path = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: snapshot_download(
                    repo_id=model_name,
                    cache_dir=str(cache_path),
                    local_dir=str(download_path),
                    revision=revision,
                    local_dir_use_symlinks=False
                )
            )
            
            # Calculate model size
            model_size = self._calculate_directory_size(Path(local_path))
            size_gb = model_size / (1024 ** 3)
            
            # Update model entry
            self.registry.update_model_size(model_id, size_gb)
            self.registry.update_model_status(model_id, ModelStatus.AVAILABLE)
            
            # Mark progress as complete
            progress.complete()
            self._notify_progress(model_id, {
                'model_id': model_id,
                'progress_percent': 100.0,
                'status': 'completed',
                'size_gb': size_gb
            })
            
            logger.info("Model download completed", 
                       model_id=model_id, 
                       size_gb=size_gb,
                       path=str(local_path))
            
            return True
            
        except Exception as e:
            logger.error("Model download failed", model_id=model_id, error=str(e))
            
            # Update progress and registry
            if model_id in self.active_downloads:
                self.active_downloads[model_id].error(str(e))
            
            self.registry.update_model_status(model_id, ModelStatus.ERROR)
            return False
            
        finally:
            # Cleanup
            if model_id in self.active_downloads:
                del self.active_downloads[model_id]
            if model_id in self.download_callbacks:
                del self.download_callbacks[model_id]

    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory in bytes"""
        total_size = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size

    def get_download_progress(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get download progress for a model"""
        if model_id not in self.active_downloads:
            return None
            
        progress = self.active_downloads[model_id]
        return {
            'model_id': model_id,
            'progress_percent': progress.progress_percent,
            'downloaded_mb': progress.downloaded_size / (1024 * 1024),
            'total_mb': progress.total_size / (1024 * 1024),
            'speed_mbps': progress.download_speed,
            'status': progress.status,
            'elapsed_seconds': (datetime.now() - progress.start_time).total_seconds(),
            'eta_seconds': (progress.total_size - progress.downloaded_size) / (progress.download_speed * 1024 * 1024) if progress.download_speed > 0 else None
        }
    
    def list_active_downloads(self) -> Dict[str, Dict[str, Any]]:
        """List all active downloads with progress"""
        result = {}
        for model_id in self.active_downloads:
            progress = self.get_download_progress(model_id)
            if progress is not None:
                result[model_id] = progress
        return result
    
    def cancel_download(self, model_id: str) -> bool:
        """Cancel an active download"""
        if model_id not in self.active_downloads:
            return False
            
        try:
            # Mark as cancelled
            self.active_downloads[model_id].error("cancelled")
            
            # Update registry
            self.registry.update_model_status(model_id, ModelStatus.ERROR)
            
            logger.info("Download cancelled", model_id=model_id)
            return True
            
        except Exception as e:
            logger.error("Failed to cancel download", model_id=model_id, error=str(e))
            return False

    def delete_model(self, model_id: str) -> bool:
        """Delete a downloaded model"""
        try:
            # Get model info
            model = self.registry.get_model(model_id)
            if not model:
                logger.warning("Model not found in registry", model_id=model_id)
                return False
            
            # Delete files
            model_path = self.download_dir / model_id
            if model_path.exists():
                shutil.rmtree(model_path)
                logger.info("Model files deleted", model_id=model_id, path=str(model_path))
            
            # Remove from registry
            success = self.registry.unregister_model(model_id)
            if success:
                logger.info("Model removed from registry", model_id=model_id)
            
            return success
            
        except Exception as e:
            logger.error("Failed to delete model", model_id=model_id, error=str(e))
            return False
    
    async def search_huggingface_models(self, 
                                      query: str,
                                      model_type: Optional[str] = None,
                                      sort: str = "downloads",
                                      limit: int = 20) -> List[Dict[str, Any]]:
        """Search for models on HuggingFace Hub"""
        try:
            # Prepare search parameters
            search_kwargs = {
                'search': query,
                'sort': sort,
                'direction': -1,  # Descending
                'limit': limit
            }
            
            # Add pipeline tag if model_type is specified
            if model_type:
                # Map our model types to HF pipeline tags
                type_mapping = {
                    'llm': 'text-generation',
                    'vision': 'image-classification',
                    'tts': 'text-to-speech',
                    'diffusion': 'text-to-image',
                    'embedding': 'feature-extraction'
                }
                if model_type.lower() in type_mapping:
                    search_kwargs['pipeline_tag'] = type_mapping[model_type.lower()]
            
            # Search models using list_models
            models = list(list_models(**search_kwargs))
            
            # Format results
            results = []
            for model in models:
                # Get model ID - the attribute name may vary between versions
                model_id = getattr(model, 'id', getattr(model, 'modelId', str(model)))
                
                results.append({
                    'id': model_id,
                    'name': model_id.split('/')[-1] if '/' in model_id else model_id,
                    'author': model_id.split('/')[0] if '/' in model_id else 'unknown',
                    'downloads': getattr(model, 'downloads', 0),
                    'likes': getattr(model, 'likes', 0),
                    'tags': getattr(model, 'tags', []),
                    'pipeline_tag': getattr(model, 'pipeline_tag', ''),
                    'created_at': getattr(model, 'created_at', None),
                    'last_modified': getattr(model, 'last_modified', None)
                })
            
            return results
            
        except Exception as e:
            logger.error("Failed to search HuggingFace models", query=query, error=str(e))
            return []

    def cleanup(self):
        """Cleanup resources"""
        # Cancel all active downloads
        for model_id in list(self.active_downloads.keys()):
            self.cancel_download(model_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("ModelDownloader cleanup completed")
