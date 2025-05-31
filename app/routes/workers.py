"""
Worker management API routes
Handles worker registration, health monitoring, and assignment operations
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query

from ..core.dependencies import get_model_service
from ..services.model_service import ModelService
from ..schemas.models import (
    WorkerInfo, WorkerListResponse, WorkerSearchParams, 
    ApiResponse, WorkerStatus
)

router = APIRouter(prefix="/workers", tags=["workers"])


@router.get("/", response_model=WorkerListResponse)
async def list_workers(
    status: Optional[str] = Query(None, description="Filter by worker status"),
    gpu_index: Optional[int] = Query(None, ge=0, le=4, description="Filter by GPU index"),
    min_memory_gb: Optional[float] = Query(None, ge=0, description="Minimum available memory"),
    has_models: Optional[bool] = Query(None, description="Filter workers with/without loaded models"),
    model_service: ModelService = Depends(get_model_service)
):
    """List workers with filtering"""
    try:
        # Build search parameters
        params = WorkerSearchParams(
            status=WorkerStatus(status) if status else None,
            gpu_index=gpu_index,
            min_memory_gb=min_memory_gb,
            has_models=has_models
        )
        
        workers, total = await model_service.list_workers(params)
        online_count = sum(1 for w in workers if w.status == WorkerStatus.ONLINE)
        
        return WorkerListResponse(
            workers=workers,
            total=total,
            online_count=online_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{worker_id}", response_model=WorkerInfo)
async def get_worker(
    worker_id: str,
    model_service: ModelService = Depends(get_model_service)
):
    """Get worker by ID"""
    try:
        worker = await model_service.get_worker(worker_id)
        if not worker:
            raise HTTPException(status_code=404, detail="Worker not found")
        return worker
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=WorkerInfo)
async def register_worker(
    worker: WorkerInfo,
    model_service: ModelService = Depends(get_model_service)
):
    """Register a new worker"""
    try:
        registered_worker = await model_service.register_worker(worker)
        return registered_worker
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{worker_id}", response_model=WorkerInfo)
async def update_worker(
    worker_id: str,
    updates: dict,
    model_service: ModelService = Depends(get_model_service)
):
    """Update worker information"""
    try:
        updated_worker = await model_service.update_worker(worker_id, updates)
        if not updated_worker:
            raise HTTPException(status_code=404, detail="Worker not found")
        return updated_worker
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{worker_id}/heartbeat", response_model=ApiResponse)
async def worker_heartbeat(
    worker_id: str,
    model_service: ModelService = Depends(get_model_service)
):
    """Update worker heartbeat"""
    try:
        updated_worker = await model_service.update_worker_heartbeat(worker_id)
        if not updated_worker:
            raise HTTPException(status_code=404, detail="Worker not found")
        
        return ApiResponse(
            success=True,
            message=f"Heartbeat updated for worker {worker_id}",
            data=None,
            error=None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{worker_id}", response_model=ApiResponse)
async def delete_worker(
    worker_id: str,
    model_service: ModelService = Depends(get_model_service)
):
    """Delete worker and unassign all models"""
    try:
        deleted = await model_service.delete_worker(worker_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Worker not found")
        
        return ApiResponse(
            success=True,
            message=f"Worker {worker_id} deleted successfully",
            data=None,
            error=None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{worker_id}/optimal-for/{model_id}")
async def check_worker_optimal_for_model(
    worker_id: str,
    model_id: str,
    model_service: ModelService = Depends(get_model_service)
):
    """Check if worker is optimal for a specific model"""
    try:
        worker = await model_service.get_worker(worker_id)
        if not worker:
            raise HTTPException(status_code=404, detail="Worker not found")
        
        model = await model_service.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        optimal_worker = await model_service.find_optimal_worker(model_id)
        is_optimal = optimal_worker and optimal_worker.id == worker_id
        
        return {
            "worker_id": worker_id,
            "model_id": model_id,
            "is_optimal": is_optimal,
            "worker_status": worker.status,
            "can_load_model": worker.can_load_model,
            "memory_usage_percent": worker.memory_usage_percent,
            "loaded_models_count": len(worker.loaded_models)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup-stale", response_model=ApiResponse)
async def cleanup_stale_workers(
    timeout_minutes: int = Query(10, ge=1, le=60, description="Timeout in minutes"),
    model_service: ModelService = Depends(get_model_service)
):
    """Remove workers that haven't sent heartbeats recently"""
    try:
        removed_count = await model_service.cleanup_stale_workers(timeout_minutes)
        
        return ApiResponse(
            success=True,
            message=f"Removed {removed_count} stale workers",
            data={"removed_count": removed_count},
            error=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
