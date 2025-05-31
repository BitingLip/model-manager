# Model Manager Development Guide

## Development Setup

### Prerequisites
- Python 3.10 or higher
- Git for version control
- Virtual environment tool (venv, conda, or poetry)

### Local Development Environment

1. **Clone and Navigate**:
```bash
cd model-manager
```

2. **Create Virtual Environment**:
```bash
# Using venv
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Using conda
conda create -n model-manager python=3.10
conda activate model-manager
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

4. **Configure Environment**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Run Tests**:
```bash
pytest tests/ -v
```

### Dependencies

#### Core Dependencies (`requirements.txt`)
```txt
huggingface-hub>=0.19.0    # HuggingFace model downloads
pydantic>=2.0.0           # Data validation and settings
structlog>=23.1.0         # Structured logging
aiofiles>=23.1.0          # Async file operations
aiohttp>=3.8.0            # Async HTTP client
fastapi>=0.104.0          # API framework (optional)
uvicorn>=0.24.0           # ASGI server (optional)
```

#### Development Dependencies (`requirements-dev.txt`)
```txt
pytest>=7.4.0             # Testing framework
pytest-asyncio>=0.21.0    # Async testing support
pytest-cov>=4.1.0         # Coverage reporting
black>=23.0.0             # Code formatting
isort>=5.12.0             # Import sorting
flake8>=6.0.0             # Linting
mypy>=1.5.0               # Type checking
pre-commit>=3.4.0         # Git hooks
```

## Project Structure

```
model-manager/
├── app/                          # Application code
│   ├── models/                   # Core model management
│   │   ├── __init__.py
│   │   ├── downloader.py         # Model downloading logic
│   │   ├── registry.py           # Model registry management
│   │   ├── schemas.py            # Data models and schemas
│   │   └── exceptions.py         # Custom exceptions
│   ├── config.py                 # Configuration management
│   ├── logging.py                # Logging setup
│   └── main.py                   # Entry point (if running as service)
├── docs/                         # Documentation
├── models/                       # Model storage directory
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── fixtures/                 # Test fixtures
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
├── pyproject.toml               # Project configuration
└── README.md                    # Project overview
```

## Coding Standards

### Code Style

**Formatting**: Use `black` for code formatting
```bash
black app/ tests/
```

**Import Sorting**: Use `isort` for import organization
```bash
isort app/ tests/
```

**Linting**: Use `flake8` for code linting
```bash
flake8 app/ tests/
```

**Type Checking**: Use `mypy` for static type checking
```bash
mypy app/
```

### Code Quality Tools Configuration

#### `pyproject.toml`
```toml
[tool.black]
line-length = 88
target-version = ['py310']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["app"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
show_error_codes = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short --strict-markers"
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests"
]
```

### Pre-commit Hooks

Install pre-commit hooks to ensure code quality:

```bash
pre-commit install
```

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [pydantic, types-aiofiles]
```

## Testing Strategy

### Test Structure

```
tests/
├── unit/                         # Unit tests
│   ├── test_registry.py          # Test model registry
│   ├── test_downloader.py        # Test model downloader
│   └── test_schemas.py           # Test data schemas
├── integration/                  # Integration tests
│   ├── test_hf_integration.py    # HuggingFace integration
│   └── test_full_workflow.py     # End-to-end workflows
├── fixtures/                     # Test fixtures and mocks
│   ├── mock_models.py            # Mock model data
│   └── test_models/              # Sample model files
└── conftest.py                   # Pytest configuration
```

### Unit Testing

**Registry Tests** (`tests/unit/test_registry.py`):
```python
import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from app.models.registry import ModelRegistry
from app.models.schemas import ModelEntry, ModelType, ModelStatus


@pytest.fixture
def temp_registry():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        registry_file = Path(f.name)
    
    registry = ModelRegistry(registry_file=registry_file)
    yield registry
    
    # Cleanup
    if registry_file.exists():
        registry_file.unlink()


def test_add_model(temp_registry):
    """Test adding a new model to registry."""
    entry = ModelEntry(
        model_id="test-model",
        model_type=ModelType.TEXT_GENERATION,
        path=Path("/tmp/test-model"),
        status=ModelStatus.AVAILABLE,
        version="1.0",
        size_bytes=1000,
        download_date=datetime.now(),
        metadata={}
    )
    
    result = temp_registry.add_model(entry)
    assert result is True
    
    retrieved = temp_registry.get_model("test-model")
    assert retrieved is not None
    assert retrieved.model_id == "test-model"


def test_duplicate_model(temp_registry):
    """Test adding duplicate model returns False."""
    entry = ModelEntry(
        model_id="duplicate-test",
        model_type=ModelType.TEXT_GENERATION,
        path=Path("/tmp/duplicate"),
        status=ModelStatus.AVAILABLE,
        version="1.0",
        size_bytes=1000,
        download_date=datetime.now(),
        metadata={}
    )
    
    # First add should succeed
    assert temp_registry.add_model(entry) is True
    
    # Second add should fail
    assert temp_registry.add_model(entry) is False
```

**Downloader Tests** (`tests/unit/test_downloader.py`):
```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

from app.models.downloader import ModelDownloader
from app.models.registry import ModelRegistry
from app.models.schemas import ModelType, ModelStatus


@pytest.fixture
def mock_registry():
    return MagicMock(spec=ModelRegistry)


@pytest.fixture
def downloader(mock_registry):
    return ModelDownloader(
        registry=mock_registry,
        download_dir=Path("/tmp/models"),
        max_workers=1
    )


@pytest.mark.asyncio
async def test_download_model_hf_success(downloader, mock_registry):
    """Test successful model download from HuggingFace."""
    model_id = "test/model"
    
    # Mock registry methods
    mock_registry.get_model.return_value = None
    mock_registry.add_model.return_value = True
    mock_registry.update_model_status.return_value = True
    
    with patch('app.models.downloader.snapshot_download') as mock_download:
        mock_download.return_value = "/tmp/models/test/model"
        
        result = await downloader.download_model_hf(
            model_id=model_id,
            model_type=ModelType.TEXT_GENERATION
        )
        
        assert result is not None
        mock_registry.add_model.assert_called_once()
        mock_registry.update_model_status.assert_called()


@pytest.mark.asyncio
async def test_download_existing_model(downloader, mock_registry):
    """Test downloading already existing model."""
    model_id = "existing/model"
    
    # Mock existing model
    existing_entry = MagicMock()
    existing_entry.status = ModelStatus.AVAILABLE
    mock_registry.get_model.return_value = existing_entry
    
    result = await downloader.download_model_hf(
        model_id=model_id,
        model_type=ModelType.TEXT_GENERATION
    )
    
    assert result == existing_entry
    mock_registry.add_model.assert_not_called()
```

### Integration Testing

**HuggingFace Integration** (`tests/integration/test_hf_integration.py`):
```python
import pytest
import tempfile
from pathlib import Path

from app.models.registry import ModelRegistry
from app.models.downloader import ModelDownloader
from app.models.schemas import ModelType, ModelStatus


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_hf_download():
    """Test downloading a real small model from HuggingFace."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        registry_file = temp_path / "registry.json"
        
        registry = ModelRegistry(registry_file=registry_file)
        downloader = ModelDownloader(
            registry=registry,
            download_dir=temp_path / "models"
        )
        
        # Download a small test model
        result = await downloader.download_model_hf(
            model_id="hf-internal-testing/tiny-random-gpt2",
            model_type=ModelType.TEXT_GENERATION
        )
        
        assert result is not None
        assert result.status == ModelStatus.AVAILABLE
        assert result.path.exists()
        assert any(result.path.iterdir())  # Has files
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run only unit tests
pytest tests/unit/ -m unit

# Run only integration tests
pytest tests/integration/ -m integration

# Run specific test file
pytest tests/unit/test_registry.py -v

# Run specific test function
pytest tests/unit/test_registry.py::test_add_model -v
```

## Development Workflow

### 1. Feature Development

1. **Create Feature Branch**:
```bash
git checkout -b feature/model-validation
```

2. **Implement Feature**:
   - Write failing tests first (TDD approach)
   - Implement the feature
   - Ensure all tests pass
   - Update documentation

3. **Code Quality Checks**:
```bash
# Format code
black app/ tests/
isort app/ tests/

# Run linting
flake8 app/ tests/

# Type checking
mypy app/

# Run tests
pytest --cov=app
```

4. **Commit Changes**:
```bash
git add .
git commit -m "feat: add model validation support

- Add checksum validation for downloaded models
- Implement file integrity checks
- Add validation configuration options
- Update tests and documentation"
```

### 2. Code Review Process

1. **Push Feature Branch**:
```bash
git push origin feature/model-validation
```

2. **Create Pull Request** with:
   - Clear description of changes
   - Link to related issues
   - Test coverage information
   - Documentation updates

3. **Review Checklist**:
   - [ ] Code follows style guidelines
   - [ ] Tests pass and provide good coverage
   - [ ] Documentation is updated
   - [ ] No breaking changes (or properly documented)
   - [ ] Performance implications considered

### 3. Release Process

1. **Version Bumping**:
```bash
# Update version in pyproject.toml
# Update CHANGELOG.md
git tag v1.2.0
git push origin v1.2.0
```

2. **Release Notes**:
   - Document new features
   - List bug fixes
   - Note breaking changes
   - Include migration guide if needed

## Debugging and Troubleshooting

### Logging Configuration

Configure structured logging for better debugging:

```python
# app/logging.py
import structlog
import logging
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: Path = None):
    """Configure structured logging."""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.dev.ConsoleRenderer(colors=True)
    ]
    
    if log_file:
        # Add file logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        )
        logging.getLogger().addHandler(file_handler)
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
```

### Common Issues and Solutions

**Issue: Download Progress Not Updating**
```python
# Enable debug logging for downloads
import logging
logging.getLogger("huggingface_hub").setLevel(logging.DEBUG)

# Check progress tracking implementation
progress = downloader.get_download_progress("model_id")
logger.info("Download progress", progress=progress)
```

**Issue: Model Files Corrupted**
```python
# Verify file integrity
import hashlib

def verify_file_integrity(file_path: Path, expected_hash: str) -> bool:
    """Verify file using SHA256 hash."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == expected_hash
```

**Issue: Registry Corruption**
```python
# Backup and repair registry
def backup_and_repair_registry(registry: ModelRegistry):
    """Create backup and attempt registry repair."""
    import shutil
    import json
    
    registry_file = registry.registry_file
    backup_file = registry_file.with_suffix('.json.backup')
    
    # Create backup
    if registry_file.exists():
        shutil.copy2(registry_file, backup_file)
    
    # Attempt repair
    try:
        registry.load()
    except json.JSONDecodeError:
        logger.error("Registry corrupted, restoring from backup")
        if backup_file.exists():
            shutil.copy2(backup_file, registry_file)
            registry.load()
```

## Performance Optimization

### Async Best Practices

```python
# Use async context managers for resource management
async def download_with_semaphore(semaphore, model_id, model_type):
    async with semaphore:
        return await downloader.download_model_hf(model_id, model_type)

# Limit concurrent downloads
import asyncio
semaphore = asyncio.Semaphore(3)

tasks = [
    download_with_semaphore(semaphore, model_id, model_type)
    for model_id, model_type in models_to_download
]

results = await asyncio.gather(*tasks)
```

### Memory Management

```python
# Use generators for large file operations
def read_file_chunks(file_path: Path, chunk_size: int = 8192):
    """Read file in chunks to manage memory usage."""
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            yield chunk

# Implement LRU cache for registry lookups
from functools import lru_cache

class ModelRegistry:
    @lru_cache(maxsize=1000)
    def get_model_cached(self, model_id: str) -> Optional[ModelEntry]:
        """Cached model lookup for frequently accessed models."""
        return self._get_model_internal(model_id)
```

### Monitoring and Metrics

```python
# Add performance metrics
import time
from functools import wraps

def track_performance(func):
    """Decorator to track function performance."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            logger.info(
                "Function performance",
                function=func.__name__,
                duration=duration,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys())
            )
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            logger.info(
                "Function performance",
                function=func.__name__,
                duration=duration
            )
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# Usage
@track_performance
async def download_model_hf(self, model_id: str, model_type: ModelType):
    # Implementation
    pass
```
