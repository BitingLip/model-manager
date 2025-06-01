"""
Simple Registry Service Adapter
Provides async interface to the existing ModelRegistry
"""

from typing import List, Optional, Union
import asyncio
import logging

from ..schemas.models import ModelEntry, WorkerInfo
from ..models.registry import ModelRegistry
from ..models.postgresql_registry import PostgreSQLModelRegistry

logger = logging.getLogger(__name__)

# Type alias for any registry implementation
RegistryType = Union[ModelRegistry, PostgreSQLModelRegistry]


class RegistryService:
    """Adapter service for ModelRegistry with async interface"""
    
    def __init__(self, registry: RegistryType):
        self.registry = registry
        logger.info("RegistryService adapter initialized")

    def _is_postgresql_registry(self) -> bool:
        """Check if using PostgreSQL registry"""
        return hasattr(self.registry, '_get_connection')  # PostgreSQL specific method

    # Model operations
    async def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Get model by ID"""
        def _get():
            return self.registry.get_model(model_id)
        return await asyncio.to_thread(_get)
    
    async def list_models(self, search_term: Optional[str] = None) -> List[ModelEntry]:
        """List models with optional search"""
        def _list():
            if self._is_postgresql_registry():
                # PostgreSQL registry returns list directly, no search parameter
                return self.registry.list_models()
            else:
                # SQLite registry returns tuple (models, pagination) and supports search
                models, _ = self.registry.list_models(search=search_term)
                return models
        return await asyncio.to_thread(_list)
    
    async def add_model(self, model: ModelEntry) -> ModelEntry:
        """Add/register a new model"""
        def _add():
            # Both registries use register_model method
            self.registry.register_model(model)
            return model
        return await asyncio.to_thread(_add)
    
    async def update_model(self, model_id: str, model: ModelEntry) -> Optional[ModelEntry]:
        """Update existing model"""
        def _update():
            existing = self.registry.get_model(model_id)
            if not existing:
                return None
            
            # Update the model's ID to match the path parameter
            model.id = model_id
            
            if self._is_postgresql_registry():
                # PostgreSQL registry: update_model(model_id: str, updates: Dict[str, Any])
                updates = {
                    'name': model.name,
                    'type': model.type.value,
                    'size_gb': model.size_gb,
                    'status': model.status.value,
                    'assigned_worker': model.assigned_worker,
                    'download_progress': model.download_progress,
                    'description': model.description,
                    'tags': model.tags,
                    'capabilities': model.capabilities,
                    'requirements': model.requirements,
                    'avg_inference_time': model.avg_inference_time,
                    'usage_count': model.usage_count
                }
                success = self.registry.update_model(model_id, updates)
            else:
                # SQLite registry: update_model(model: ModelEntry)
                success = self.registry.update_model(model)
            
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
            if self._is_postgresql_registry():
                # PostgreSQL registry returns list directly
                return self.registry.list_workers()
            else:
                # SQLite registry returns tuple (workers, pagination)
                workers, _ = self.registry.list_workers()
                return workers
        return await asyncio.to_thread(_list)
    
    async def register_worker(self, worker: WorkerInfo) -> WorkerInfo:
        """Register new worker"""
        def _register():
            # Both registries use register_worker method
            self.registry.register_worker(worker)
            return worker
        return await asyncio.to_thread(_register)
    
    async def update_worker(self, worker_id: str, worker: WorkerInfo) -> Optional[WorkerInfo]:
        """Update existing worker"""
        def _update():
            existing = self.registry.get_worker(worker_id)
            if not existing:
                return None
            
            # Update the worker's ID to match the path parameter
            worker.id = worker_id
            
            if self._is_postgresql_registry():
                # PostgreSQL registry: update_worker(worker_id: str, updates: Dict[str, Any])
                updates = {
                    'gpu_index': worker.gpu_index,
                    'hostname': worker.hostname,
                    'memory_total_gb': worker.memory_total_gb,
                    'memory_used_gb': worker.memory_used_gb,
                    'memory_available_gb': worker.memory_available_gb,
                    'loaded_models': worker.loaded_models,
                    'max_models': worker.max_models,
                    'status': worker.status.value,
                    'error_message': worker.error_message
                }
                success = self.registry.update_worker(worker_id, updates)
            else:
                # SQLite registry: update_worker(worker: WorkerInfo)
                success = self.registry.update_worker(worker)
            
            return worker if success else None
        return await asyncio.to_thread(_update)

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
            if self._is_postgresql_registry():
                updates = {'assigned_worker': worker_id}
                return self.registry.update_model(model_id, updates)
            else:
                model.assigned_worker = worker_id
                return self.registry.update_model(model)
        return await asyncio.to_thread(_assign)
    
    async def unassign_model_from_worker(self, model_id: str) -> bool:
        """Remove model assignment from worker"""
        def _unassign():
            model = self.registry.get_model(model_id)
            if not model:
                return False
            
            if self._is_postgresql_registry():
                updates = {'assigned_worker': None}
                return self.registry.update_model(model_id, updates)
            else:
                model.assigned_worker = None
                return self.registry.update_model(model)
        return await asyncio.to_thread(_unassign)
    
    async def get_worker_models(self, worker_id: str) -> List[ModelEntry]:
        """Get models assigned to a specific worker"""
        def _get():
            if self._is_postgresql_registry():
                # PostgreSQL registry - need to filter manually since no assigned_worker parameter
                all_models = self.registry.list_models()
                return [m for m in all_models if m.assigned_worker == worker_id]
            else:
                # SQLite registry supports assigned_worker filter
                models, _ = self.registry.list_models(assigned_worker=worker_id)
                return models
        return await asyncio.to_thread(_get)
