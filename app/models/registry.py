"""
Model Registry - SQLite-based storage for models and workers
Provides thread-safe CRUD operations and search functionality
"""

import sqlite3
import json
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from contextlib import contextmanager

import structlog
from ..schemas.models import ModelEntry, WorkerInfo, ModelType, ModelStatus, WorkerStatus

logger = structlog.get_logger(__name__)

class ModelRegistry:
    """Thread-safe SQLite-based model and worker registry"""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize registry with database path"""
        self.db_path = db_path or Path("model_registry.db")
        self._lock = threading.Lock()
        self._init_database()
        
        logger.info("ModelRegistry initialized", db_path=str(self.db_path))
    
    def _init_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            # Models table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    size_gb REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'available',
                    assigned_worker TEXT,
                    download_progress REAL DEFAULT 0.0,
                    description TEXT,
                    tags TEXT,  -- JSON array
                    capabilities TEXT,  -- JSON array
                    requirements TEXT,  -- JSON object
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_used TEXT,
                    avg_inference_time REAL DEFAULT 0.0,
                    usage_count INTEGER DEFAULT 0
                )
            """)
            
            # Workers table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workers (
                    id TEXT PRIMARY KEY,
                    gpu_index INTEGER NOT NULL,
                    hostname TEXT NOT NULL,
                    memory_total_gb REAL NOT NULL,
                    memory_used_gb REAL DEFAULT 0.0,
                    memory_available_gb REAL NOT NULL,
                    loaded_models TEXT DEFAULT '[]',  -- JSON array of model IDs
                    max_models INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'offline',
                    last_heartbeat TEXT NOT NULL,
                    error_message TEXT,
                    avg_load_time REAL DEFAULT 0.0,
                    total_inferences INTEGER DEFAULT 0
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_models_type ON models(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_models_status ON models(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_models_worker ON models(assigned_worker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_workers_status ON workers(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_workers_gpu ON workers(gpu_index)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get thread-safe database connection"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row  # Enable column access by name
            try:
                yield conn
            finally:
                conn.close()
    
    def _row_to_model(self, row: sqlite3.Row) -> ModelEntry:
        """Convert database row to ModelEntry"""
        # Convert row to dict
        data = dict(row)
        
        # Parse JSON fields
        data['tags'] = json.loads(data['tags']) if data['tags'] else []
        data['capabilities'] = json.loads(data['capabilities']) if data['capabilities'] else []
        data['requirements'] = json.loads(data['requirements']) if data['requirements'] else {}
        
        # Convert enum fields
        data['type'] = ModelType(data['type'])
        data['status'] = ModelStatus(data['status'])
        
        # Parse datetime fields
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if data['last_used']:
            data['last_used'] = datetime.fromisoformat(data['last_used'])
        
        return ModelEntry(**data)
    
    def _row_to_worker(self, row: sqlite3.Row) -> WorkerInfo:
        """Convert database row to WorkerInfo"""
        # Convert row to dict
        data = dict(row)
        
        # Parse JSON fields
        data['loaded_models'] = json.loads(data['loaded_models']) if data['loaded_models'] else []
        
        # Convert enum fields
        data['status'] = WorkerStatus(data['status'])
        
        # Parse datetime fields
        data['last_heartbeat'] = datetime.fromisoformat(data['last_heartbeat'])
        
        return WorkerInfo(**data)
    
    # Model CRUD Operations
    def register_model(self, model: ModelEntry) -> bool:
        """Register a new model"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO models (
                        id, name, type, size_gb, status, assigned_worker, download_progress,
                        description, tags, capabilities, requirements, created_at, updated_at,
                        last_used, avg_inference_time, usage_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model.id, model.name, model.type.value, model.size_gb, model.status.value,
                    model.assigned_worker, model.download_progress, model.description,
                    json.dumps(model.tags), json.dumps(model.capabilities),
                    json.dumps(model.requirements), model.created_at.isoformat(),
                    model.updated_at.isoformat(),
                    model.last_used.isoformat() if model.last_used else None,
                    model.avg_inference_time, model.usage_count
                ))
                conn.commit()
                
            logger.info("Model registered successfully", model_id=model.id, model_name=model.name)
            return True
            
        except sqlite3.IntegrityError as e:
            logger.error("Model registration failed - ID already exists", model_id=model.id, error=str(e))
            return False
        except Exception as e:
            logger.error("Model registration failed", model_id=model.id, error=str(e))
            return False
    
    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Get model by ID"""
        try:
            with self._get_connection() as conn:
                row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
                return self._row_to_model(row) if row else None
                
        except Exception as e:
            logger.error("Failed to get model", model_id=model_id, error=str(e))
            return None
    
    def update_model(self, model: ModelEntry) -> bool:
        """Update existing model"""
        try:
            with self._get_connection() as conn:
                conn.execute("""                    UPDATE models SET 
                        name = ?, type = ?, size_gb = ?, status = ?, assigned_worker = ?,
                        download_progress = ?, description = ?, tags = ?, capabilities = ?,
                        requirements = ?, updated_at = ?, last_used = ?,
                        avg_inference_time = ?, usage_count = ?
                    WHERE id = ?
                """, (
                    model.name, model.type.value, model.size_gb, model.status.value,
                    model.assigned_worker, model.download_progress, model.description,
                    json.dumps(model.tags), json.dumps(model.capabilities),
                    json.dumps(model.requirements), model.updated_at.isoformat(),
                    model.last_used.isoformat() if model.last_used else None,
                    model.avg_inference_time, model.usage_count, model.id
                ))
                
                if conn.total_changes == 0:
                    logger.warning("Model update failed - model not found", model_id=model.id)
                    return False
                    
                conn.commit()
                
            logger.debug("Model updated successfully", model_id=model.id)
            return True
            
        except Exception as e:
            logger.error("Model update failed", model_id=model.id, error=str(e))
            return False
    
    def delete_model(self, model_id: str) -> bool:
        """Delete model"""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
                
                if conn.total_changes == 0:
                    logger.warning("Model deletion failed - model not found", model_id=model_id)
                    return False
                    
                conn.commit()
                
            logger.info("Model deleted successfully", model_id=model_id)
            return True
            
        except Exception as e:
            logger.error("Model deletion failed", model_id=model_id, error=str(e))
            return False
    
    def list_models(self, 
                   model_type: Optional[ModelType] = None,
                   status: Optional[ModelStatus] = None,
                   assigned_worker: Optional[str] = None,
                   tag: Optional[str] = None,
                   search: Optional[str] = None,
                   page: int = 1,
                   page_size: int = 50) -> Tuple[List[ModelEntry], int]:
        """List models with filtering and pagination"""
        try:
            # Build WHERE clause
            where_conditions = []
            params = []
            
            if model_type:
                where_conditions.append("type = ?")
                params.append(model_type if isinstance(model_type, str) else model_type.value)
            
            if status:
                where_conditions.append("status = ?")
                params.append(status if isinstance(status, str) else status.value)
            
            if assigned_worker:
                where_conditions.append("assigned_worker = ?")
                params.append(assigned_worker)
            
            if tag:
                where_conditions.append("tags LIKE ?")
                params.append(f'%"{tag}"%')
            
            if search:
                where_conditions.append("(name LIKE ? OR description LIKE ?)")
                params.extend([f"%{search}%", f"%{search}%"])
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            with self._get_connection() as conn:
                # Get total count
                count_query = f"SELECT COUNT(*) FROM models WHERE {where_clause}"
                total_count = conn.execute(count_query, params).fetchone()[0]
                
                # Get paginated results
                offset = (page - 1) * page_size
                query = f"""
                    SELECT * FROM models WHERE {where_clause} 
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """
                rows = conn.execute(query, params + [page_size, offset]).fetchall()
                models = [self._row_to_model(row) for row in rows]
                
            return models, total_count
            
        except Exception as e:
            logger.error("Failed to list models", error=str(e))
            return [], 0
    
    def update_model_status(self, model_id: str, status: ModelStatus) -> bool:
        """Update model status"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE models SET status = ?, updated_at = ?
                    WHERE id = ?
                """, (status.value, datetime.utcnow().isoformat(), model_id))
                
                if conn.total_changes == 0:
                    logger.warning("Model status update failed - model not found", model_id=model_id)
                    return False
                    
                conn.commit()
                
            logger.debug("Model status updated successfully", model_id=model_id, status=status.value)
            return True
            
        except Exception as e:
            logger.error("Model status update failed", model_id=model_id, error=str(e))
            return False
    
    def update_model_progress(self, model_id: str, progress: float) -> bool:
        """Update model download progress"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE models SET download_progress = ?, updated_at = ?
                    WHERE id = ?
                """, (progress, datetime.utcnow().isoformat(), model_id))
                
                if conn.total_changes == 0:
                    logger.warning("Model progress update failed - model not found", model_id=model_id)
                    return False
                    
                conn.commit()
                
            logger.debug("Model progress updated successfully", model_id=model_id, progress=progress)
            return True
            
        except Exception as e:
            logger.error("Model progress update failed", model_id=model_id, error=str(e))
            return False
    
    def update_model_size(self, model_id: str, size_gb: float) -> bool:
        """Update model size"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE models SET size_gb = ?, updated_at = ?
                    WHERE id = ?
                """, (size_gb, datetime.utcnow().isoformat(), model_id))
                
                if conn.total_changes == 0:
                    logger.warning("Model size update failed - model not found", model_id=model_id)
                    return False
                    
                conn.commit()
                
            logger.debug("Model size updated successfully", model_id=model_id, size_gb=size_gb)
            return True
            
        except Exception as e:
            logger.error("Model size update failed", model_id=model_id, error=str(e))
            return False
    
    # Worker CRUD Operations
    def register_worker(self, worker: WorkerInfo) -> bool:
        """Register or update worker"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO workers (
                        id, gpu_index, hostname, memory_total_gb, memory_used_gb,
                        memory_available_gb, loaded_models, max_models, status,
                        last_heartbeat, error_message, avg_load_time, total_inferences
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (                    worker.id, worker.gpu_index, worker.hostname, worker.memory_total_gb,
                    worker.memory_used_gb, worker.memory_available_gb,
                    json.dumps(worker.loaded_models), worker.max_models, worker.status.value,
                    worker.last_heartbeat.isoformat(), worker.error_message,
                    worker.avg_load_time, worker.total_inferences
                ))
                conn.commit()
                
            logger.info("Worker registered successfully", worker_id=worker.id, gpu_index=worker.gpu_index)
            return True
            
        except Exception as e:
            logger.error("Worker registration failed", worker_id=worker.id, error=str(e))
            return False
    
    def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get worker by ID"""
        try:
            with self._get_connection() as conn:
                row = conn.execute("SELECT * FROM workers WHERE id = ?", (worker_id,)).fetchone()
                return self._row_to_worker(row) if row else None
                
        except Exception as e:
            logger.error("Failed to get worker", worker_id=worker_id, error=str(e))
            return None
    
    def update_worker(self, worker: WorkerInfo) -> bool:
        """Update existing worker"""
        try:
            with self._get_connection() as conn:
                conn.execute("""                    UPDATE workers SET
                        gpu_index = ?, hostname = ?, memory_total_gb = ?, memory_used_gb = ?,
                        memory_available_gb = ?, loaded_models = ?, max_models = ?, status = ?,
                        last_heartbeat = ?, error_message = ?, avg_load_time = ?, total_inferences = ?
                    WHERE id = ?
                """, (                    worker.gpu_index, worker.hostname, worker.memory_total_gb,
                    worker.memory_used_gb, worker.memory_available_gb,
                    json.dumps(worker.loaded_models), worker.max_models, worker.status.value,
                    worker.last_heartbeat.isoformat(), worker.error_message,worker.avg_load_time, worker.total_inferences, worker.id
                ))
                
                if conn.total_changes == 0:
                    logger.warning("Worker update failed - worker not found", worker_id=worker.id)
                    return False
                    
                conn.commit()
                
            logger.debug("Worker updated successfully", worker_id=worker.id)
            return True
            
        except Exception as e:
            logger.error("Worker update failed", worker_id=worker.id, error=str(e))
            return False
    
    def list_workers(self, 
                    status: Optional[WorkerStatus] = None,
                    gpu_index: Optional[int] = None,
                    min_memory_gb: Optional[float] = None,
                    has_models: Optional[bool] = None) -> Tuple[List[WorkerInfo], int]:
        """List workers with filtering"""
        try:
            where_conditions = []
            params = []
            
            if status:
                where_conditions.append("status = ?")
                params.append(status if isinstance(status, str) else status.value)
            
            if gpu_index is not None:
                where_conditions.append("gpu_index = ?")
                params.append(gpu_index)
            
            if min_memory_gb is not None:
                where_conditions.append("memory_available_gb >= ?")
                params.append(min_memory_gb)
            
            if has_models is not None:
                if has_models:
                    where_conditions.append("loaded_models != '[]' AND loaded_models IS NOT NULL")
                else:
                    where_conditions.append("(loaded_models = '[]' OR loaded_models IS NULL)")
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            with self._get_connection() as conn:
                # Get total count
                count_query = f"SELECT COUNT(*) FROM workers WHERE {where_clause}"
                total_count = conn.execute(count_query, params).fetchone()[0]
                
                # Get workers
                query = f"SELECT * FROM workers WHERE {where_clause} ORDER BY gpu_index"
                rows = conn.execute(query, params).fetchall()
                workers = [self._row_to_worker(row) for row in rows]
                
            return workers, total_count
            
        except Exception as e:
            logger.error("Failed to list workers", error=str(e))
            return [], 0
    
    # Utility methods
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            with self._get_connection() as conn:
                # Model statistics
                model_count = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
                worker_count = conn.execute("SELECT COUNT(*) FROM workers").fetchone()[0]
                
                # Model status distribution
                model_status_rows = conn.execute("""
                    SELECT status, COUNT(*) as count 
                    FROM models 
                    GROUP BY status
                """).fetchall()
                model_status_counts = {row[0]: row[1] for row in model_status_rows}
                
                # Worker status distribution
                worker_status_rows = conn.execute("""
                    SELECT status, COUNT(*) as count 
                    FROM workers 
                    GROUP BY status
                """).fetchall()
                worker_status_counts = {row[0]: row[1] for row in worker_status_rows}
                
                return {
                    'total_models': model_count,
                    'total_workers': worker_count,
                    'model_status_counts': model_status_counts,
                    'worker_status_counts': worker_status_counts
                }
                
        except Exception as e:
            logger.error("Failed to get system stats", error=str(e))
            return {}
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            with self._get_connection() as conn:
                # Model statistics
                model_stats = {}
                cursor = conn.execute("SELECT status, COUNT(*) as count FROM models GROUP BY status")
                for row in cursor:
                    model_stats[row['status']] = row['count']
                
                total_models = conn.execute("SELECT COUNT(*) as count FROM models").fetchone()['count']
                
                # Worker statistics  
                worker_stats = {}
                cursor = conn.execute("SELECT status, COUNT(*) as count FROM workers GROUP BY status")
                for row in cursor:
                    worker_stats[row['status']] = row['count']
                
                total_workers = conn.execute("SELECT COUNT(*) as count FROM workers").fetchone()['count']
                
                # Storage statistics
                cursor = conn.execute("SELECT SUM(size_gb) as total_size FROM models WHERE status != 'removed'")
                total_storage = cursor.fetchone()['total_size'] or 0.0
                
                return {
                    "total_models": total_models,
                    "total_workers": total_workers,
                    "model_status_counts": model_stats,
                    "worker_status_counts": worker_stats,
                    "total_storage_gb": round(total_storage, 2),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error("Failed to get system statistics", error=str(e))
            return {
                "total_models": 0,
                "total_workers": 0,
                "model_status_counts": {},
                "worker_status_counts": {},
                "total_storage_gb": 0.0,
                "error": str(e)
            }
    
    def cleanup_orphaned_assignments(self) -> int:
        """Clean up models assigned to non-existent workers"""
        try:
            with self._get_connection() as conn:
                # Find models assigned to non-existent workers
                result = conn.execute("""
                    UPDATE models 
                    SET assigned_worker = NULL, status = 'available'
                    WHERE assigned_worker IS NOT NULL 
                    AND assigned_worker NOT IN (SELECT id FROM workers WHERE status != 'offline')
                """)
                
                cleaned_count = result.rowcount
                conn.commit()
                
            if cleaned_count > 0:
                logger.info("Cleaned up orphaned model assignments", count=cleaned_count)
                
            return cleaned_count
            
        except Exception as e:
            logger.error("Failed to cleanup orphaned assignments", error=str(e))
            return 0
