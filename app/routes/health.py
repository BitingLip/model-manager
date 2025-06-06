"""
Health and system status API routes
Provides system monitoring, health checks, and statistics
"""

from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from typing import Any
from typing import Any

from ..core.dependencies import get_model_service
# from ..services.model_service import ModelService  # Temporarily commented out
from ..schemas.models import SystemStatusResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check(
    model_service: Any = Depends(get_model_service)  # Use Any for now to accept MinimalModelService
):
    """Basic health check endpoint"""
    try:
        # Use system status to create a basic health check
        system_status = await model_service.get_system_status()
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "total_models": system_status.get('models', {}).get('total', 0),
            "database_connected": True  # If we get here, DB is working
        }
        return health_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system", response_model=SystemStatusResponse)
async def get_system_status(
    model_service: Any = Depends(get_model_service)  # Use Any for now
):
    """Get comprehensive system status"""
    try:
        # Use the ModelService get_system_status method
        system_status = await model_service.get_system_status()
        
        # Transform to match SystemStatusResponse schema
        models_info = system_status.get('models', {})
        workers_info = system_status.get('workers', {})
        
        return {
            "total_models": models_info.get('total', 0),
            "available_models": models_info.get('available', 0),
            "downloading_models": models_info.get('downloading', 0), 
            "loaded_models": 0,  # Not available in current system status
            "total_workers": workers_info.get('total', 0),
            "online_workers": workers_info.get('online', 0),
            "busy_workers": workers_info.get('busy', 0),
            "total_memory_gb": 0.0,  # Not available in current system status
            "used_memory_gb": 0.0,   # Not available in current system status
            "memory_usage_percent": 0.0,  # Not available in current system status
            "system_healthy": True,  # Basic health check
            "issues": []  # No issues tracking yet
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_system_statistics(
    model_service: Any = Depends(get_model_service)  # Use Any for now
):
    """Get detailed system statistics"""
    try:
        model_stats = await model_service.get_model_statistics()
        workers, total_workers = await model_service.list_workers()
        
        # Worker statistics
        worker_stats = {
            'total': total_workers,
            'by_status': {},
            'by_gpu': {},
            'total_memory_gb': 0.0,
            'used_memory_gb': 0.0,
            'avg_memory_usage': 0.0,
            'most_loaded': None
        }
        
        for worker in workers:
            # By status
            status = worker.status.value
            worker_stats['by_status'][status] = worker_stats['by_status'].get(status, 0) + 1
            
            # By GPU
            gpu = f"gpu_{worker.gpu_index}"
            worker_stats['by_gpu'][gpu] = worker_stats['by_gpu'].get(gpu, 0) + 1
            
            # Memory statistics
            worker_stats['total_memory_gb'] += worker.memory_total_gb
            worker_stats['used_memory_gb'] += worker.memory_used_gb
        
        if total_workers > 0:
            worker_stats['avg_memory_usage'] = (
                worker_stats['used_memory_gb'] / worker_stats['total_memory_gb'] * 100
                if worker_stats['total_memory_gb'] > 0 else 0.0
            )
            
            # Most loaded worker
            most_loaded = max(workers, key=lambda w: len(w.loaded_models), default=None)
            if most_loaded:
                worker_stats['most_loaded'] = {
                    'id': most_loaded.id,
                    'gpu_index': most_loaded.gpu_index,
                    'loaded_models_count': len(most_loaded.loaded_models),
                    'memory_usage_percent': most_loaded.memory_usage_percent
                }
        
        return {
            'models': model_stats,
            'workers': worker_stats,
            'system_healthy': len(workers) > 0 and model_stats['total'] > 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/readiness")
async def readiness_check(
    model_service: Any = Depends(get_model_service)  # Use Any for now
):
    """Kubernetes readiness probe endpoint"""
    try:
        status = await model_service.get_system_status()
        
        # Basic readiness criteria
        ready = (
            status.total_workers > 0 and  # At least one worker
            status.online_workers > 0 and  # At least one online worker
            status.memory_usage_percent < 95  # Memory not critically full
        )
        
        if ready:
            return {
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat(),
                "online_workers": status.online_workers,
                "memory_usage": status.memory_usage_percent
            }
        else:
            raise HTTPException(
                status_code=503, 
                detail={
                    "status": "not_ready",
                    "issues": status.issues,
                    "online_workers": status.online_workers,
                    "memory_usage": status.memory_usage_percent
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/liveness")
async def liveness_check():
    """Kubernetes liveness probe endpoint"""
    return {"status": "alive", "service": "model-manager"}


@router.get("/metrics")
async def get_metrics(
    model_service: Any = Depends(get_model_service)  # Use Any for now
):
    """Get metrics in a format suitable for monitoring systems"""
    try:
        status = await model_service.get_system_status()
        model_stats = await model_service.get_model_statistics()
        
        metrics = {
            # Model metrics
            "model_manager_models_total": status.total_models,
            "model_manager_models_available": status.available_models,
            "model_manager_models_downloading": status.downloading_models,
            "model_manager_models_loaded": status.loaded_models,
            "model_manager_models_size_gb": model_stats.get('total_size_gb', 0),
            
            # Worker metrics
            "model_manager_workers_total": status.total_workers,
            "model_manager_workers_online": status.online_workers,
            "model_manager_workers_busy": status.busy_workers,
            
            # Memory metrics
            "model_manager_memory_total_gb": status.total_memory_gb,
            "model_manager_memory_used_gb": status.used_memory_gb,
            "model_manager_memory_usage_percent": status.memory_usage_percent,
            
            # System health
            "model_manager_system_healthy": 1 if status.system_healthy else 0,
            "model_manager_issues_count": len(status.issues)
        }
        
        return {"metrics": metrics, "timestamp": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
