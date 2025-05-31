"""
Simple Registry Service Adapter
Provides async interface to the existing ModelRegistry
"""

from typing import List, Optional
import asyncio
import logging

from ..schemas.models import ModelEntry, WorkerInfo
from ..models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class RegistryService:
    """Adapter service for ModelRegistry with async interface"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        logger.info("RegistryService adapter initialized")
      # Model operations
    async def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Get model by ID"""
        def _get():
            return self.registry.get_model(model_id)
        return await asyncio.to_thread(_get)
    
    async def list_models(self, search_term: Optional[str] = None) -> List[ModelEntry]:
        """List models with optional search"""
        def _list():
            models, _ = self.registry.list_models(search=search_term)  # Fixed parameter name and unpacked tuple
            return models
        return await asyncio.to_thread(_list)
    
    async def add_model(self, model: ModelEntry) -> ModelEntry:
        """Add/register a new model"""
        def _add():
            self.registry.register_model(model)  # Uses register_model, not add_model
            return model
        await asyncio.to_thread(_add)
        return model
    
    async def update_model(self, model_id: str, model: ModelEntry) -> Optional[ModelEntry]:
        """Update existing model - model parameter expects ModelEntry object"""
        def _update():
            existing = self.registry.get_model(model_id)
            if not existing:
                return None
            # Update the model's ID to match the path parameter
            model.id = model_id
            success = self.registry.update_model(model)  # Passes ModelEntry object directly
            return model if success else None
        return await asyncio.to_thread(_update)
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete model from registry"""
        def _delete():
            return self.registry.delete_model(model_id)
        return await asyncio.to_thread(_delete)
      # Worker operations
    async def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get worker by ID"""
        def _get():
            return self.registry.get_worker(worker_id)
        return await asyncio.to_thread(_get)
    
    async def list_workers(self) -> List[WorkerInfo]:
        """List all workers"""
        def _list():
            workers, _ = self.registry.list_workers()  # Unpack tuple return value
            return workers
        return await asyncio.to_thread(_list)
    
    async def register_worker(self, worker: WorkerInfo) -> WorkerInfo:
        """Register new worker"""
        def _register():
            self.registry.register_worker(worker)
            return worker
        await asyncio.to_thread(_register)
        return worker
    
    async def update_worker(self, worker_id: str, worker: WorkerInfo) -> Optional[WorkerInfo]:
        """Update existing worker - worker parameter expects WorkerInfo object"""
        def _update():
            existing = self.registry.get_worker(worker_id)
            if not existing:
                return None
            # Update the worker's ID to match the path parameter
            worker.id = worker_id
            success = self.registry.update_worker(worker)  # Passes WorkerInfo object directly
            return worker if success else None
        return await asyncio.to_thread(_update)
      # Note: The following methods don't exist in ModelRegistry, so removing them
    # - unregister_worker
    # - assign_model_to_worker  
    # - get_worker_models
    # These would need to be implemented at the database level if needed
    
    # Additional functionality using existing methods
    async def assign_model_to_worker(self, model_id: str, worker_id: str) -> bool:
        """Assign model to worker by updating model's assigned_worker field"""
        def _assign():
            model = self.registry.get_model(model_id)
            if not model:
                return False
            worker = self.registry.get_worker(worker_id)
            if not worker:
                return False
            # Update model to assign it to the worker
            model.assigned_worker = worker_id
            return self.registry.update_model(model)
        return await asyncio.to_thread(_assign)
    
    async def unassign_model_from_worker(self, model_id: str) -> bool:
        """Remove model assignment from worker"""
        def _unassign():
            model = self.registry.get_model(model_id)
            if not model:
                return False
            model.assigned_worker = None
            return self.registry.update_model(model)
        return await asyncio.to_thread(_unassign)
    
    async def get_worker_models(self, worker_id: str) -> List[ModelEntry]:
        """Get models assigned to a specific worker"""
        def _get():
            models, _ = self.registry.list_models(assigned_worker=worker_id)
            return models
        return await asyncio.to_thread(_get)
