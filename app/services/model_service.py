"""
Simple Model Service
High-level service orchestrating model management operations
"""

from typing import Optional, List, Dict, Any, Tuple
import logging
from datetime import datetime

from ..schemas.models import (
    ModelEntry, WorkerInfo, ModelStatus, WorkerStatus,
    ModelDownloadRequest, WorkerSearchParams, ModelSearchParams
)
from .registry_service import RegistryService
from .download_service import DownloadService

logger = logging.getLogger(__name__)


class ModelService:
    """High-level service orchestrating model management operations"""
    
    def __init__(self, registry_service: RegistryService, download_service: DownloadService):
        """Initialize model service with adapter dependencies"""
        self.registry_service = registry_service
        self.download_service = download_service
        logger.info("ModelService initialized")
      # Model Management
    async def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Get model by ID"""
        return await self.registry_service.get_model(model_id)
        
    async def list_models(self, params: Optional[ModelSearchParams] = None) -> Tuple[List[ModelEntry], int]:
        """List models with filtering and pagination"""
        # Convert ModelSearchParams to search term
        search_term = params.search if params else None
        models = await self.registry_service.list_models(search_term=search_term)
        
        # Apply filtering (could be enhanced in the future)
        if params and (params.model_type or params.status or params.assigned_worker or params.tag):
            filtered = []
            for model in models:
                if params.model_type and model.type != params.model_type:
                    continue
                if params.status and model.status != params.status:
                    continue
                if params.assigned_worker and model.assigned_worker != params.assigned_worker:
                    continue
                if params.tag and params.tag not in model.tags:
                    continue
                filtered.append(model)
            models = filtered
            
        return models, len(models)
    
    async def download_model(self, request: ModelDownloadRequest) -> ModelEntry:
        """Download and register a new model"""
        try:
            logger.info("Starting model download")
            
            # First download the model
            model = await self.download_service.download_model(request)
            
            # Then register it in the registry
            registered_model = await self.registry_service.add_model(model)
            
            logger.info("Model download completed")
            return registered_model
            
        except Exception as e:
            logger.error("Failed to download model")
            raise
    
    async def delete_model(self, model_id: str, delete_files: bool = True) -> bool:
        """Delete model and optionally its files"""
        try:
            model = await self.registry_service.get_model(model_id)
            if not model:
                return False
            
            # Delete files if requested
            if delete_files:
                await self.download_service.delete_model_files(model_id)
            
            # Remove from registry
            deleted = await self.registry_service.delete_model(model_id)
            
            if deleted:
                logger.info("Model deleted")
            
            return deleted
        except Exception as e:
            logger.error("Failed to delete model")
            raise
    
    async def get_download_progress(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get download progress for a model"""
        return await self.download_service.get_download_progress(model_id)
    
    async def cancel_download(self, model_id: str) -> bool:
        """Cancel an active download"""
        return await self.download_service.cancel_download(model_id)
    
    async def search_huggingface_models(self, query: str, model_type: Optional[str] = None, 
                                       limit: int = 20) -> List[Dict[str, Any]]:
        """Search for models on HuggingFace"""
        return await self.download_service.search_huggingface_models(query, model_type, limit)
    
    async def add_model(self, model: ModelEntry) -> ModelEntry:
        """Register a new model directly"""
        return await self.registry_service.add_model(model)
    
    # Worker Management
    async def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get worker by ID"""
        return await self.registry_service.get_worker(worker_id)
    
    async def list_workers(self, params: Optional[WorkerSearchParams] = None) -> Tuple[List[WorkerInfo], int]:
        """List workers with filtering"""
        # Currently ignoring params as the underlying registry doesn't support filtering
        # This will need to be enhanced when filtering is supported
        workers = await self.registry_service.list_workers()
        # Manual filtering since the registry doesn't support it
        filtered = workers
        if params:
            filtered = []
            for worker in workers:
                if params.status and worker.status != params.status:
                    continue
                if params.gpu_index is not None and worker.gpu_index != params.gpu_index:
                    continue
                if params.min_memory_gb is not None and worker.memory_available_gb < params.min_memory_gb:
                    continue
                if params.has_models is not None:
                    has_loaded = len(worker.loaded_models) > 0
                    if params.has_models != has_loaded:
                        continue
                filtered.append(worker)
                
        return filtered, len(filtered)
    
    async def register_worker(self, worker: WorkerInfo) -> WorkerInfo:
        """Register a new worker"""
        return await self.registry_service.register_worker(worker)
    
    async def update_worker(self, worker_id: str, worker: WorkerInfo) -> Optional[WorkerInfo]:
        """Update worker information"""
        return await self.registry_service.update_worker(worker_id, worker)
    
    # Note: unregister_worker removed - not supported by ModelRegistry
    # Workers can be marked as offline but not removed from registry
    
    # Model Assignment
    async def assign_model(self, model_id: str, worker_id: str) -> bool:
        """Assign a model to a worker"""
        try:
            # Verify model exists
            model = await self.registry_service.get_model(model_id)
            if not model:
                logger.error(f"Model not found for assignment: {model_id}")
                return False
            
            # Verify worker exists
            worker = await self.registry_service.get_worker(worker_id)
            if not worker:
                logger.error(f"Worker not found for assignment: {worker_id}")
                return False
            
            # Assign model to worker
            success = await self.registry_service.assign_model_to_worker(model_id, worker_id)
            
            if success:
                logger.info(f"Model {model_id} assigned to worker {worker_id}")
            
            return success
        except Exception as e:
            logger.error(f"Failed to assign model {model_id} to worker {worker_id}: {str(e)}")
            return False
    
    async def get_worker_models(self, worker_id: str) -> List[ModelEntry]:
        """Get all models assigned to a worker"""
        return await self.registry_service.get_worker_models(worker_id)
    
    # System Status
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        try:
            models = await self.list_models()
            workers = await self.list_workers()
            downloads = await self.download_service.list_downloads()
            
            # Calculate statistics
            model_stats = {
                'total': len(models),
                'downloading': len([m for m in models if m.status == ModelStatus.DOWNLOADING]),
                'available': len([m for m in models if m.status == ModelStatus.AVAILABLE]),
                'error': len([m for m in models if m.status == ModelStatus.ERROR]),
                'total_size_gb': sum(m.size_gb for m in models)
            }
            
            worker_stats = {
                'total': len(workers),
                'online': len([w for w in workers if w.status == WorkerStatus.ONLINE]),
                'busy': len([w for w in workers if w.status == WorkerStatus.BUSY]),
                'offline': len([w for w in workers if w.status == WorkerStatus.OFFLINE])
            }
            
            download_stats = {
                'active_downloads': len(downloads)
            }
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'models': model_stats,
                'workers': worker_stats,
                'downloads': download_stats
            }
        except Exception as e:
            logger.error("Failed to get system status")
            raise
    
    def cleanup(self):
        """Cleanup service resources"""
        try:
            self.download_service.cleanup()
            logger.info("ModelService cleanup completed")
        except Exception as e:
            logger.error("Failed to cleanup ModelService")
