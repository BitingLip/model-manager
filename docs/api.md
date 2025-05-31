# Model Manager API Reference

## ModelRegistry Class

### Overview
The `ModelRegistry` class manages model metadata and provides CRUD operations for model entries.

### Initialization
```python
from app.models.registry import ModelRegistry
from pathlib import Path

registry = ModelRegistry(
    registry_file=Path("./models/registry.json")
)
```

**Parameters**:
- `registry_file` (Path): Path to the JSON file storing model metadata

### Methods

#### `add_model(entry: ModelEntry) -> bool`
Adds a new model entry to the registry.

**Parameters**:
- `entry` (ModelEntry): Model entry object to add

**Returns**:
- `bool`: True if added successfully, False if model already exists

**Example**:
```python
from app.models.schemas import ModelEntry, ModelType, ModelStatus
from datetime import datetime
from pathlib import Path

entry = ModelEntry(
    model_id="gpt2",
    model_type=ModelType.TEXT_GENERATION,
    path=Path("./models/huggingface/gpt2"),
    status=ModelStatus.DOWNLOADING,
    version="main",
    size_bytes=0,
    download_date=datetime.now(),
    metadata={"source": "huggingface"}
)

success = registry.add_model(entry)
```

#### `get_model(model_id: str) -> Optional[ModelEntry]`
Retrieves a model entry by ID.

**Parameters**:
- `model_id` (str): Unique identifier for the model

**Returns**:
- `Optional[ModelEntry]`: Model entry if found, None otherwise

**Example**:
```python
model = registry.get_model("gpt2")
if model:
    print(f"Model status: {model.status}")
    print(f"Model path: {model.path}")
```

#### `list_models(model_type: Optional[ModelType] = None, status: Optional[ModelStatus] = None) -> List[ModelEntry]`
Lists models with optional filtering.

**Parameters**:
- `model_type` (Optional[ModelType]): Filter by model type
- `status` (Optional[ModelStatus]): Filter by model status

**Returns**:
- `List[ModelEntry]`: List of matching model entries

**Examples**:
```python
# List all models
all_models = registry.list_models()

# List only text generation models
text_models = registry.list_models(model_type=ModelType.TEXT_GENERATION)

# List only available models
available_models = registry.list_models(status=ModelStatus.AVAILABLE)

# List available text generation models
available_text_models = registry.list_models(
    model_type=ModelType.TEXT_GENERATION,
    status=ModelStatus.AVAILABLE
)
```

#### `update_model_status(model_id: str, status: ModelStatus, metadata: Optional[Dict] = None) -> bool`
Updates the status and optionally metadata of a model.

**Parameters**:
- `model_id` (str): Model identifier
- `status` (ModelStatus): New status
- `metadata` (Optional[Dict]): Additional metadata to merge

**Returns**:
- `bool`: True if updated successfully, False if model not found

**Example**:
```python
# Update status to available
registry.update_model_status("gpt2", ModelStatus.AVAILABLE)

# Update with additional metadata
registry.update_model_status(
    "gpt2", 
    ModelStatus.AVAILABLE,
    metadata={"download_duration": 120, "file_count": 5}
)
```

#### `remove_model(model_id: str) -> bool`
Removes a model from the registry.

**Parameters**:
- `model_id` (str): Model identifier

**Returns**:
- `bool`: True if removed successfully, False if model not found

**Example**:
```python
removed = registry.remove_model("old-model")
if removed:
    print("Model removed from registry")
```

#### `save() -> None`
Persists the current registry state to disk.

**Example**:
```python
registry.save()  # Manually save changes
```

#### `load() -> None`
Loads the registry state from disk.

**Example**:
```python
registry.load()  # Reload from file
```

## ModelDownloader Class

### Overview
The `ModelDownloader` class handles downloading models from various sources, primarily HuggingFace Hub.

### Initialization
```python
from app.models.downloader import ModelDownloader
from app.models.registry import ModelRegistry
from pathlib import Path

registry = ModelRegistry(registry_file=Path("./models/registry.json"))
downloader = ModelDownloader(
    registry=registry,
    download_dir=Path("./models"),
    max_workers=3
)
```

**Parameters**:
- `registry` (ModelRegistry): Registry instance for metadata management
- `download_dir` (Path): Base directory for model downloads
- `max_workers` (int, optional): Maximum concurrent downloads (default: 3)

### Methods

#### `async download_model_hf(model_id: str, model_type: ModelType, revision: str = "main", specific_files: Optional[List[str]] = None) -> Optional[ModelEntry]`
Downloads a model from HuggingFace Hub.

**Parameters**:
- `model_id` (str): HuggingFace model identifier (e.g., "gpt2", "openai/whisper-base")
- `model_type` (ModelType): Type classification for the model
- `revision` (str, optional): Git revision/branch to download (default: "main")
- `specific_files` (Optional[List[str]]): List of specific files to download (default: all files)

**Returns**:
- `Optional[ModelEntry]`: Model entry if successful, None if failed

**Examples**:
```python
import asyncio

async def download_models():
    # Download GPT-2 model
    gpt2_entry = await downloader.download_model_hf(
        model_id="gpt2",
        model_type=ModelType.TEXT_GENERATION
    )
    
    # Download specific revision
    specific_revision = await downloader.download_model_hf(
        model_id="openai/whisper-base",
        model_type=ModelType.AUDIO_TO_TEXT,
        revision="v1.0"
    )
    
    # Download only specific files
    config_only = await downloader.download_model_hf(
        model_id="bert-base-uncased",
        model_type=ModelType.EMBEDDING,
        specific_files=["config.json", "tokenizer.json"]
    )

asyncio.run(download_models())
```

#### `async get_model_info_hf(model_id: str) -> Optional[Dict]`
Retrieves model information from HuggingFace Hub without downloading.

**Parameters**:
- `model_id` (str): HuggingFace model identifier

**Returns**:
- `Optional[Dict]`: Model metadata if successful, None if failed

**Example**:
```python
async def check_model_info():
    info = await downloader.get_model_info_hf("gpt2")
    if info:
        print(f"Model size: {info.get('size', 'Unknown')}")
        print(f"Model tags: {info.get('tags', [])}")

asyncio.run(check_model_info())
```

#### `get_download_progress(model_id: str) -> Optional[Dict]`
Gets current download progress for a model.

**Parameters**:
- `model_id` (str): Model identifier

**Returns**:
- `Optional[Dict]`: Progress information if download in progress, None otherwise

**Progress Dictionary**:
```python
{
    "model_id": "gpt2",
    "status": "downloading",
    "progress": 0.75,  # 0.0 to 1.0
    "downloaded_bytes": 750000000,
    "total_bytes": 1000000000,
    "files_completed": 3,
    "files_total": 5,
    "estimated_remaining": 30  # seconds
}
```

**Example**:
```python
progress = downloader.get_download_progress("gpt2")
if progress:
    percentage = progress["progress"] * 100
    print(f"Download progress: {percentage:.1f}%")
```

#### `cancel_download(model_id: str) -> bool`
Cancels an ongoing download.

**Parameters**:
- `model_id` (str): Model identifier

**Returns**:
- `bool`: True if cancelled successfully, False if no download in progress

**Example**:
```python
cancelled = downloader.cancel_download("large-model")
if cancelled:
    print("Download cancelled successfully")
```

## Data Schemas

### ModelEntry
```python
@dataclass
class ModelEntry:
    model_id: str              # Unique identifier
    model_type: ModelType      # Model type classification
    path: Path                 # Local storage path
    status: ModelStatus        # Current status
    version: str               # Model version/revision
    size_bytes: int            # Total size in bytes
    download_date: datetime    # When downloaded
    metadata: Dict[str, Any]   # Additional metadata
```

### ModelType Enumeration
```python
class ModelType(str, Enum):
    TEXT_GENERATION = "text-generation"    # GPT, LLaMA, etc.
    TEXT_TO_IMAGE = "text-to-image"        # Stable Diffusion, DALL-E
    IMAGE_TO_TEXT = "image-to-text"        # BLIP, CLIP
    AUDIO_TO_TEXT = "audio-to-text"        # Whisper, Wav2Vec
    TEXT_TO_AUDIO = "text-to-audio"        # TTS models
    EMBEDDING = "embedding"                # BERT, Sentence Transformers
    CLASSIFICATION = "classification"       # Classification models
    OTHER = "other"                        # Other model types
```

### ModelStatus Enumeration
```python
class ModelStatus(str, Enum):
    DOWNLOADING = "downloading"    # Currently downloading
    AVAILABLE = "available"        # Ready for use
    ERROR = "error"               # Download or validation error
    DELETED = "deleted"           # Marked for deletion
    PENDING = "pending"           # Queued for download
```

## Error Handling

### Common Exceptions

#### `ModelDownloadError`
Raised when model download fails.

```python
from app.models.exceptions import ModelDownloadError

try:
    await downloader.download_model_hf("invalid/model", ModelType.TEXT_GENERATION)
except ModelDownloadError as e:
    print(f"Download failed: {e}")
    print(f"Error code: {e.error_code}")
    print(f"Retry after: {e.retry_after}")
```

#### `ModelRegistryError`
Raised when registry operations fail.

```python
from app.models.exceptions import ModelRegistryError

try:
    registry.add_model(duplicate_entry)
except ModelRegistryError as e:
    print(f"Registry error: {e}")
```

### Error Codes
- `NETWORK_ERROR`: Network connectivity issues
- `AUTHENTICATION_ERROR`: Invalid HuggingFace token
- `NOT_FOUND`: Model not found on source
- `INSUFFICIENT_SPACE`: Not enough disk space
- `PERMISSION_ERROR`: File system permission issues
- `CHECKSUM_MISMATCH`: Downloaded file integrity check failed

## Usage Examples

### Complete Download Workflow
```python
import asyncio
from pathlib import Path
from app.models.registry import ModelRegistry
from app.models.downloader import ModelDownloader
from app.models.schemas import ModelType

async def download_workflow():
    # Initialize components
    storage_path = Path("./models")
    registry_file = storage_path / "registry.json"
    
    registry = ModelRegistry(registry_file=registry_file)
    downloader = ModelDownloader(
        registry=registry,
        download_dir=storage_path,
        max_workers=2
    )
    
    # Check if model exists
    model_id = "microsoft/DialoGPT-medium"
    existing_model = registry.get_model(model_id)
    
    if existing_model:
        print(f"Model {model_id} already available at {existing_model.path}")
        return existing_model
    
    # Download new model
    print(f"Downloading {model_id}...")
    model_entry = await downloader.download_model_hf(
        model_id=model_id,
        model_type=ModelType.TEXT_GENERATION
    )
    
    if model_entry and model_entry.status == ModelStatus.AVAILABLE:
        print(f"Download completed: {model_entry.path}")
        print(f"Model size: {model_entry.size_bytes / (1024**2):.1f} MB")
        return model_entry
    else:
        print("Download failed")
        return None

# Run the workflow
result = asyncio.run(download_workflow())
```

### Batch Model Management
```python
async def batch_download():
    models_to_download = [
        ("gpt2", ModelType.TEXT_GENERATION),
        ("openai/whisper-base", ModelType.AUDIO_TO_TEXT),
        ("sentence-transformers/all-MiniLM-L6-v2", ModelType.EMBEDDING)
    ]
    
    tasks = []
    for model_id, model_type in models_to_download:
        task = downloader.download_model_hf(model_id, model_type)
        tasks.append(task)
    
    # Download all models concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for i, result in enumerate(results):
        model_id = models_to_download[i][0]
        if isinstance(result, Exception):
            print(f"Failed to download {model_id}: {result}")
        else:
            print(f"Successfully downloaded {model_id}")

asyncio.run(batch_download())
```
