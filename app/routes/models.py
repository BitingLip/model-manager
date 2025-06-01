"""
Model management API routes
Handles model download, assignment, and lifecycle operations
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse

from ..core.dependencies import get_model_service
from ..services.model_service import ModelService
from ..schemas.models import (
    ModelEntry, ModelDownloadRequest, ModelAssignRequest, 
    ModelListResponse, ModelSearchParams, ApiResponse,
    ModelType, ModelStatus
)

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/", response_model=ModelListResponse)
async def list_models(
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    status: Optional[str] = Query(None, description="Filter by model status"),
    assigned_worker: Optional[str] = Query(None, description="Filter by assigned worker"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    search: Optional[str] = Query(None, description="Search in name and description"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Page size"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    model_service: ModelService = Depends(get_model_service)
):
    """List models with filtering and pagination"""
    try:        # Build search parameters
        params = ModelSearchParams(
            model_type=ModelType(model_type) if model_type else None,
            status=ModelStatus(status) if status else None,
            assigned_worker=assigned_worker,
            tag=tag,
            search=search,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        models, total = await model_service.list_models(params)
        
        return ModelListResponse(
            models=models,
            total=total,
            page=page,
            page_size=page_size
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}", response_model=ModelEntry)
async def get_model(
    model_id: str,
    model_service: ModelService = Depends(get_model_service)
):
    """Get model by ID"""
    try:
        model = await model_service.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        return model
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/download", response_model=ModelEntry)
async def download_model(
    request: ModelDownloadRequest,
    background_tasks: BackgroundTasks,
    model_service: ModelService = Depends(get_model_service)
):
    """Download a model from HuggingFace"""
    try:
        model = await model_service.download_model(request)
        return model
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{model_id}", response_model=ApiResponse)
async def delete_model(
    model_id: str,
    delete_files: bool = Query(True, description="Delete model files from disk"),
    model_service: ModelService = Depends(get_model_service)
):
    """Delete a model"""
    try:
        deleted = await model_service.delete_model(model_id, delete_files)
        if not deleted:
            raise HTTPException(status_code=404, detail="Model not found")        
        return ApiResponse(
            success=True,
            message=f"Model {model_id} deleted successfully",
            data=None,
            error=None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/assign", response_model=ApiResponse)
async def assign_model(
    model_id: str,
    request: Optional[ModelAssignRequest] = None,
    model_service: ModelService = Depends(get_model_service)
):
    """Assign model to a worker"""
    try:
        if not request:
            request = ModelAssignRequest(model_id=model_id, worker_id=None, force=False)
        else:
            request.model_id = model_id
            
        if not request.worker_id:
            raise HTTPException(status_code=400, detail="worker_id is required")
        assigned = await model_service.assign_model(model_id, request.worker_id)
        if not assigned:
            raise HTTPException(status_code=400, detail="Failed to assign model")
        
        return ApiResponse(
            success=True,
            message=f"Model {model_id} assigned successfully",
            data=None,
            error=None
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/unassign", response_model=ApiResponse)
async def unassign_model(
    model_id: str,
    model_service: ModelService = Depends(get_model_service)
):
    """Unassign model from worker"""
    try:
        # For now, we'll use a workaround since unassign_model doesn't exist
        # This could be implemented by assigning to None or a special worker
        raise HTTPException(status_code=501, detail="Unassign functionality not yet implemented")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}/download-progress")
async def get_download_progress(
    model_id: str,
    model_service: ModelService = Depends(get_model_service)
):
    """Get download progress for a model"""
    try:
        progress = await model_service.get_download_progress(model_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Model not found or not downloading")
        return progress
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/cancel-download", response_model=ApiResponse)
async def cancel_download(
    model_id: str,
    model_service: ModelService = Depends(get_model_service)
):
    """Cancel an active download"""
    try:
        cancelled = await model_service.cancel_download(model_id)
        if not cancelled:
            raise HTTPException(status_code=404, detail="No active download found")        
        return ApiResponse(
            success=True,
            message=f"Download cancelled for model {model_id}",
            data=None,
            error=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/huggingface")
async def search_huggingface_models(
    query: str = Query(..., description="Search query"),
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    model_service: ModelService = Depends(get_model_service)
):
    """Search for models on HuggingFace"""
    try:
        results = await model_service.search_huggingface_models(
            query, 
            ModelType(model_type) if model_type else None, 
            limit
        )
        return {"results": results, "query": query, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_model_statistics(
    model_service: ModelService = Depends(get_model_service)
):
    """Get detailed model statistics"""
    try:
        # Use system status since get_model_statistics doesn't exist
        system_status = await model_service.get_system_status()
        
        # Transform to match expected statistics format
        stats = {
            "total_models": system_status.get("models", {}).get("total", 0),
            "models_by_type": {},  # Not available in system status
            "download_stats": system_status.get("downloads", {}),
            "usage_stats": {"total_requests": 0, "avg_response_time": 0}  # Not available
        }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=ModelEntry)
async def add_model(
    model: ModelEntry,
    model_service: ModelService = Depends(get_model_service)
):
    """Add a new model directly"""
    try:
        return await model_service.add_model(model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
