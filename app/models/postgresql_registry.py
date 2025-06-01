"""
PostgreSQL Model Registry - Replaces SQLite registry
Provides thread-safe CRUD operations with PostgreSQL backend
"""

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
import json
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from contextlib import contextmanager

import structlog
from ..schemas.models import ModelEntry, WorkerInfo, ModelType, ModelStatus, WorkerStatus

logger = structlog.get_logger(__name__)

def json_serializer(obj):
    """Custom JSON serializer for datetime objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

class PostgreSQLModelRegistry:
    """Thread-safe PostgreSQL-based model and worker registry"""
    
    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        """Initialize registry with PostgreSQL connection"""
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 5432,
            'database': 'bitinglip_models',
            'user': 'model_manager',
            'password': 'model_manager_2025!'
        }
        
        # Create connection pool for thread safety
        self.pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=20,
            **self.db_config
        )
        
        logger.info("PostgreSQLModelRegistry initialized", database=self.db_config['database'])
    
    @contextmanager
    def _get_connection(self):
        """Get thread-safe database connection from pool"""
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)
    
    def _row_to_model(self, row: Dict[str, Any]) -> ModelEntry:
        """Convert database row to ModelEntry"""
        # Parse datetime fields
        if isinstance(row['created_at'], str):
            row['created_at'] = datetime.fromisoformat(row['created_at'])
        if isinstance(row['updated_at'], str):
            row['updated_at'] = datetime.fromisoformat(row['updated_at'])
        if row['last_used'] and isinstance(row['last_used'], str):
            row['last_used'] = datetime.fromisoformat(row['last_used'])
        
        # Convert enum fields
        row['type'] = ModelType(row['type'])
        row['status'] = ModelStatus(row['status'])
        
        # JSONB fields are already parsed by psycopg2
        return ModelEntry(**row)
    
    def _row_to_worker(self, row: Dict[str, Any]) -> WorkerInfo:
        """Convert database row to WorkerInfo"""
        # Parse datetime fields
        if isinstance(row['last_heartbeat'], str):
            row['last_heartbeat'] = datetime.fromisoformat(row['last_heartbeat'])
        
        # Convert enum fields
        row['status'] = WorkerStatus(row['status'])
        
        # JSONB fields are already parsed by psycopg2
        return WorkerInfo(**row)
    
    def _log_audit(self, conn, table: str, record_id: str, action: str, 
                   old_data: Optional[Dict] = None, new_data: Optional[Dict] = None):
        """Log changes to audit table"""
        audit_table = f"{table}_audit_log"
        
        with conn.cursor() as cursor:
            cursor.execute(f"""
                INSERT INTO {audit_table} (
                    {table.rstrip('s')}_id, action, old_data, new_data, changed_by
                ) VALUES (%s, %s, %s, %s, %s)            """, (
                record_id, action,
                json.dumps(old_data, default=json_serializer) if old_data else None,
                json.dumps(new_data, default=json_serializer) if new_data else None,
                'model_manager_service'
            ))
    
    # Model CRUD Operations
    def register_model(self, model: ModelEntry) -> bool:
        """Register a new model"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO models (
                            id, name, type, size_gb, status, assigned_worker, download_progress,
                            description, tags, capabilities, requirements, created_at, updated_at,
                            last_used, avg_inference_time, usage_count
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        model.id, model.name, model.type.value, model.size_gb, model.status.value,
                        model.assigned_worker, model.download_progress, model.description,
                        json.dumps(model.tags), json.dumps(model.capabilities),
                        json.dumps(model.requirements), model.created_at,
                        model.updated_at,
                        model.last_used,
                        model.avg_inference_time, model.usage_count
                    ))
                    
                    # Log audit
                    self._log_audit(conn, 'models', model.id, 'CREATE', 
                                  new_data=model.dict())
                    
                conn.commit()
                logger.info("Model registered", model_id=model.id, name=model.name)
                return True
                
        except psycopg2.IntegrityError as e:
            logger.warning("Model already exists", model_id=model.id, error=str(e))
            return False
        except Exception as e:
            logger.error("Failed to register model", model_id=model.id, error=str(e))
            return False
    
    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Get model by ID"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("SELECT * FROM models WHERE id = %s", (model_id,))
                    row = cursor.fetchone()
                    
                    if row:
                        return self._row_to_model(dict(row))
                    return None
                    
        except Exception as e:
            logger.error("Failed to get model", model_id=model_id, error=str(e))
            return None
    
    def update_model(self, model_id: str, updates: Dict[str, Any]) -> bool:
        """Update model fields"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Get current data for audit
                    cursor.execute("SELECT * FROM models WHERE id = %s", (model_id,))
                    old_row = cursor.fetchone()
                    if not old_row:
                        return False
                    old_data = dict(old_row)
                    
                    # Build update query
                    set_clauses = []
                    values = []
                    
                    for key, value in updates.items():
                        if key in ['tags', 'capabilities', 'requirements'] and isinstance(value, (list, dict)):
                            set_clauses.append(f"{key} = %s")
                            values.append(json.dumps(value))
                        else:
                            set_clauses.append(f"{key} = %s")
                            values.append(value)
                    
                    # Always update timestamp
                    set_clauses.append("updated_at = %s")
                    values.append(datetime.utcnow())
                    values.append(model_id)
                    
                    query = f"UPDATE models SET {', '.join(set_clauses)} WHERE id = %s"
                    cursor.execute(query, values)
                    
                    # Log audit
                    new_data = old_data.copy()
                    new_data.update(updates)
                    self._log_audit(conn, 'models', model_id, 'UPDATE', 
                                  old_data=old_data, new_data=new_data)
                    
                conn.commit()
                logger.info("Model updated", model_id=model_id, updates=updates)
                return True
                
        except Exception as e:
            logger.error("Failed to update model", model_id=model_id, error=str(e))
            return False
    
    def delete_model(self, model_id: str) -> bool:
        """Delete model"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Get current data for audit
                    cursor.execute("SELECT * FROM models WHERE id = %s", (model_id,))
                    old_row = cursor.fetchone()
                    if not old_row:
                        return False
                    old_data = dict(old_row)
                    
                    cursor.execute("DELETE FROM models WHERE id = %s", (model_id,))
                    
                    # Log audit
                    self._log_audit(conn, 'models', model_id, 'DELETE', 
                                  old_data=old_data)
                    
                conn.commit()
                logger.info("Model deleted", model_id=model_id)
                return True
                
        except Exception as e:
            logger.error("Failed to delete model", model_id=model_id, error=str(e))
            return False
    
    def list_models(self, model_type: Optional[ModelType] = None, 
                   status: Optional[ModelStatus] = None,
                   tags: Optional[List[str]] = None,
                   limit: int = 100, offset: int = 0) -> List[ModelEntry]:
        """List models with optional filtering"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Build query with filters
                    conditions = []
                    params = []
                    
                    if model_type:
                        conditions.append("type = %s")
                        params.append(model_type.value)
                    
                    if status:
                        conditions.append("status = %s")
                        params.append(status.value)
                    
                    if tags:
                        conditions.append("tags ?| %s")
                        params.append(tags)
                    
                    where_clause = " AND ".join(conditions)
                    if where_clause:
                        where_clause = f"WHERE {where_clause}"
                    
                    query = f"""
                        SELECT * FROM models {where_clause}
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                    """
                    params.extend([limit, offset])
                    
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    return [self._row_to_model(dict(row)) for row in rows]
                    
        except Exception as e:
            logger.error("Failed to list models", error=str(e))
            return []
    
    def search_models(self, query: str, limit: int = 20) -> List[ModelEntry]:
        """Search models by name, description, or tags"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT * FROM models 
                        WHERE name ILIKE %s 
                           OR description ILIKE %s
                           OR tags::text ILIKE %s
                        ORDER BY 
                            CASE WHEN name ILIKE %s THEN 1 ELSE 2 END,
                            created_at DESC
                        LIMIT %s
                    """, (f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%', limit))
                    
                    rows = cursor.fetchall()
                    return [self._row_to_model(dict(row)) for row in rows]
                    
        except Exception as e:
            logger.error("Failed to search models", query=query, error=str(e))
            return []
    
    # Worker CRUD Operations (similar pattern)
    def register_worker(self, worker: WorkerInfo) -> bool:
        """Register a new worker"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO workers (
                            id, gpu_index, hostname, memory_total_gb, memory_used_gb,
                            memory_available_gb, loaded_models, max_models, status,
                            last_heartbeat, error_message, avg_load_time, total_inferences
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        worker.id, worker.gpu_index, worker.hostname,
                        worker.memory_total_gb, worker.memory_used_gb,
                        worker.memory_available_gb, json.dumps(worker.loaded_models),
                        worker.max_models, worker.status.value,
                        worker.last_heartbeat, worker.error_message,
                        worker.avg_load_time, worker.total_inferences
                    ))
                    
                    # Log audit
                    self._log_audit(conn, 'workers', worker.id, 'CREATE', 
                                  new_data=worker.dict())
                    
                conn.commit()
                logger.info("Worker registered", worker_id=worker.id, hostname=worker.hostname)
                return True
                
        except psycopg2.IntegrityError as e:
            logger.warning("Worker already exists", worker_id=worker.id, error=str(e))
            return False
        except Exception as e:
            logger.error("Failed to register worker", worker_id=worker.id, error=str(e))
            return False
    
    def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get worker by ID"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("SELECT * FROM workers WHERE id = %s", (worker_id,))
                    row = cursor.fetchone()
                    
                    if row:
                        return self._row_to_worker(dict(row))
                    return None
                    
        except Exception as e:
            logger.error("Failed to get worker", worker_id=worker_id, error=str(e))
            return None
    
    def list_workers(self, status: Optional[WorkerStatus] = None,
                    limit: int = 100, offset: int = 0) -> List[WorkerInfo]:
        """List workers with optional filtering"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    if status:
                        cursor.execute("""
                            SELECT * FROM workers WHERE status = %s
                            ORDER BY last_heartbeat DESC
                            LIMIT %s OFFSET %s
                        """, (status.value, limit, offset))
                    else:
                        cursor.execute("""
                            SELECT * FROM workers
                            ORDER BY last_heartbeat DESC
                            LIMIT %s OFFSET %s
                        """, (limit, offset))
                    
                    rows = cursor.fetchall()
                    return [self._row_to_worker(dict(row)) for row in rows]
                    
        except Exception as e:
            logger.error("Failed to list workers", error=str(e))
            return []
    
    def update_worker_heartbeat(self, worker_id: str) -> bool:
        """Update worker heartbeat timestamp"""
        return self.update_worker(worker_id, {
            'last_heartbeat': datetime.utcnow(),
            'status': WorkerStatus.ONLINE.value
        })
    
    def update_worker(self, worker_id: str, updates: Dict[str, Any]) -> bool:
        """Update worker fields"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Get current data for audit
                    cursor.execute("SELECT * FROM workers WHERE id = %s", (worker_id,))
                    old_row = cursor.fetchone()
                    if not old_row:
                        return False
                    old_data = dict(old_row)
                    
                    # Build update query
                    set_clauses = []
                    values = []
                    
                    for key, value in updates.items():
                        if key == 'loaded_models' and isinstance(value, list):
                            set_clauses.append(f"{key} = %s")
                            values.append(json.dumps(value))
                        else:
                            set_clauses.append(f"{key} = %s")
                            values.append(value)
                    
                    values.append(worker_id)
                    
                    query = f"UPDATE workers SET {', '.join(set_clauses)} WHERE id = %s"
                    cursor.execute(query, values)
                    
                    # Log audit
                    new_data = old_data.copy()
                    new_data.update(updates)
                    self._log_audit(conn, 'workers', worker_id, 'UPDATE', 
                                  old_data=old_data, new_data=new_data)
                    
                conn.commit()
                return True
                
        except Exception as e:
            logger.error("Failed to update worker", worker_id=worker_id, error=str(e))
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Model statistics
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total_models,
                            COUNT(*) FILTER (WHERE status = 'available') as available_models,
                            COUNT(*) FILTER (WHERE status = 'loading') as loading_models,
                            COUNT(*) FILTER (WHERE status = 'loaded') as loaded_models,
                            SUM(size_gb) as total_size_gb,
                            AVG(avg_inference_time) as avg_inference_time
                        FROM models
                    """)
                    model_stats = dict(cursor.fetchone())
                    
                    # Worker statistics
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total_workers,
                            COUNT(*) FILTER (WHERE status = 'online') as online_workers,
                            COUNT(*) FILTER (WHERE status = 'offline') as offline_workers,
                            SUM(memory_total_gb) as total_memory_gb,
                            SUM(memory_used_gb) as used_memory_gb,
                            SUM(total_inferences) as total_inferences
                        FROM workers
                    """)
                    worker_stats = dict(cursor.fetchone())
                    
                    return {
                        'models': model_stats,
                        'workers': worker_stats,
                        'last_updated': datetime.utcnow().isoformat()
                    }
                    
        except Exception as e:
            logger.error("Failed to get statistics", error=str(e))
            return {}
    
    def close(self):
        """Close connection pool"""
        if self.pool:
            self.pool.closeall()
            logger.info("Connection pool closed")
