"""
Pydantic schemas for the model management system
Defines data structures for models, workers, and API requests/responses
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import re

class ModelType(str, Enum):
    """Supported model types"""
    LLM = "llm"
    VISION = "vision"
    TTS = "tts"
    DIFFUSION = "diffusion"
    EMBEDDING = "embedding"
    OTHER = "other"

class ModelStatus(str, Enum):
    """Model status states"""
    AVAILABLE = "available"      # Downloaded and ready
    DOWNLOADING = "downloading"  # Currently downloading
    LOADING = "loading"         # Being loaded into worker
    LOADED = "loaded"           # Loaded in worker memory
    ERROR = "error"             # Error state
    REMOVED = "removed"         # Marked for deletion

class WorkerStatus(str, Enum):
    """Worker status states"""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"

# Core Data Models
class ModelEntry(BaseModel):
    """Model registry entry"""
    id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="HuggingFace model name")
    type: ModelType = Field(..., description="Model type category")
    size_gb: float = Field(0.0, ge=0, description="Model size in GB")
    status: ModelStatus = Field(ModelStatus.AVAILABLE, description="Current model status")
    assigned_worker: Optional[str] = Field(None, description="Worker ID if loaded")
    download_progress: float = Field(0.0, ge=0, le=1.0, description="Download progress (0-1)")
    
    # Metadata
    description: Optional[str] = Field(None, description="Model description")
    tags: List[str] = Field(default_factory=list, description="Model tags")
    capabilities: List[str] = Field(default_factory=list, description="Model capabilities")
    requirements: Dict[str, Any] = Field(default_factory=dict, description="Resource requirements")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp")
    
    # Performance metrics
    avg_inference_time: Optional[float] = Field(None, ge=0, description="Average inference time in seconds")
    usage_count: int = Field(0, ge=0, description="Number of times model has been used")
    
    @validator('id')
    def validate_id(cls, v):
        """Validate model ID format"""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Model ID can only contain alphanumeric characters, hyphens, and underscores')
        return v
    
    @validator('name')
    def validate_name(cls, v):
        """Validate HuggingFace model name format"""
        if '/' in v and not re.match(r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$', v):
            raise ValueError('Invalid HuggingFace model name format')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class WorkerInfo(BaseModel):
    """Worker information"""
    id: str = Field(..., description="Worker identifier")
    gpu_index: int = Field(..., ge=0, le=4, description="GPU index (0-4)")
    hostname: str = Field(..., description="Worker hostname")
    
    # Memory information
    memory_total_gb: float = Field(..., gt=0, description="Total GPU memory in GB")
    memory_used_gb: float = Field(0.0, ge=0, description="Currently used memory in GB")
    memory_available_gb: float = Field(..., ge=0, description="Available memory in GB")
    
    # Model assignments
    loaded_models: List[str] = Field(default_factory=list, description="List of loaded model IDs")
    max_models: int = Field(3, gt=0, description="Maximum models this worker can handle")
    
    # Status and health
    status: WorkerStatus = Field(WorkerStatus.OFFLINE, description="Worker status")
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = Field(None, description="Error message if status is ERROR")
    
    # Performance metrics
    avg_load_time: Optional[float] = Field(None, ge=0, description="Average model load time in seconds")
    total_inferences: int = Field(0, ge=0, description="Total number of inferences performed")
    
    @validator('memory_used_gb')
    def validate_memory_usage(cls, v, values):
        """Ensure used memory doesn't exceed total memory"""
        if 'memory_total_gb' in values and v > values['memory_total_gb']:
            raise ValueError('Used memory cannot exceed total memory')
        return v
    
    @property
    def memory_usage_percent(self) -> float:
        """Calculate memory usage percentage"""
        if self.memory_total_gb == 0:
            return 0.0
        return (self.memory_used_gb / self.memory_total_gb) * 100
    
    @property
    def can_load_model(self) -> bool:
        """Check if worker can load another model"""
        return (len(self.loaded_models) < self.max_models and 
                self.status == WorkerStatus.ONLINE)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# API Request/Response Schemas
class ModelDownloadRequest(BaseModel):
    """Request to download a model"""
    model_name: str = Field(..., description="HuggingFace model name")
    model_type: ModelType = Field(..., description="Type of model")
    model_id: Optional[str] = Field(None, description="Custom model ID (auto-generated if not provided)")
    tags: List[str] = Field(default_factory=list, description="Tags for the model")
    
    @validator('model_id', pre=True, always=True)
    def generate_model_id(cls, v, values):
        """Auto-generate model ID if not provided"""
        if v is None and 'model_name' in values:
            # Convert model name to valid ID
            model_name = values['model_name']
            model_id = model_name.replace('/', '-').replace('_', '-').lower()
            return model_id
        return v

class ModelAssignRequest(BaseModel):
    """Request to assign a model to a worker"""
    model_id: str = Field(..., description="Model ID to assign")
    worker_id: Optional[str] = Field(None, description="Target worker ID (auto-select if not provided)")
    force: bool = Field(False, description="Force assignment even if worker is busy")

class ModelListResponse(BaseModel):
    """Response for model list requests"""
    models: List[ModelEntry] = Field(..., description="List of models")
    total: int = Field(..., description="Total number of models")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(50, description="Page size")

class WorkerListResponse(BaseModel):
    """Response for worker list requests"""
    workers: List[WorkerInfo] = Field(..., description="List of workers")
    total: int = Field(..., description="Total number of workers")
    online_count: int = Field(..., description="Number of online workers")

class SystemStatusResponse(BaseModel):
    """System status response"""
    total_models: int = Field(..., description="Total number of registered models")
    available_models: int = Field(..., description="Number of available models")
    downloading_models: int = Field(..., description="Number of models currently downloading")
    loaded_models: int = Field(..., description="Number of models currently loaded")
    
    total_workers: int = Field(..., description="Total number of workers")
    online_workers: int = Field(..., description="Number of online workers")
    busy_workers: int = Field(..., description="Number of busy workers")
    
    total_memory_gb: float = Field(..., description="Total GPU memory across all workers")
    used_memory_gb: float = Field(..., description="Currently used GPU memory")
    memory_usage_percent: float = Field(..., description="Overall memory usage percentage")
    
    system_healthy: bool = Field(..., description="Overall system health status")
    issues: List[str] = Field(default_factory=list, description="List of current issues")

class ApiResponse(BaseModel):
    """Generic API response wrapper"""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Response message")
    data: Optional[Any] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error details if success=False")

# Query Parameters
class ModelSearchParams(BaseModel):
    """Parameters for model search/filtering"""
    model_type: Optional[ModelType] = Field(None, description="Filter by model type")
    status: Optional[ModelStatus] = Field(None, description="Filter by model status")
    assigned_worker: Optional[str] = Field(None, description="Filter by assigned worker")
    tag: Optional[str] = Field(None, description="Filter by tag")
    search: Optional[str] = Field(None, description="Search in name and description")
    
    # Pagination
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(50, ge=1, le=100, description="Page size")
    
    # Sorting
    sort_by: str = Field("created_at", description="Sort field")
    sort_order: str = Field("desc", pattern="^(asc|desc)$", description="Sort order")

class WorkerSearchParams(BaseModel):
    """Parameters for worker search/filtering"""
    status: Optional[WorkerStatus] = Field(None, description="Filter by worker status")
    gpu_index: Optional[int] = Field(None, ge=0, le=4, description="Filter by GPU index")
    min_memory_gb: Optional[float] = Field(None, ge=0, description="Minimum available memory")
    has_models: Optional[bool] = Field(None, description="Filter workers with/without loaded models")
