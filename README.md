# Model Manager

🛠️ **Status: In Progress**

The Model Manager handles AI/ML model lifecycle management, storage, and distribution for the BitingLip inference platform.

## Core Features

- ✅ Centralized model storage and registry
- ✅ HuggingFace Hub model downloading  
- ✅ Model metadata and version management
- ✅ Progress tracking and checksum validation
- 🚧 API service interface (planned)
- 🚧 Caching strategies (planned)
- 🚧 Advanced security features (planned)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download a model
python -c "
from app.models.downloader import ModelDownloader
from app.models.registry import ModelRegistry
from pathlib import Path
import asyncio

async def download():
    storage_path = Path('./models')
    registry_file = storage_path / 'model_registry.json'
    registry = ModelRegistry(registry_file=registry_file)
    downloader = ModelDownloader(registry=registry, download_dir=storage_path)
    
    await downloader.download_model_hf('gpt2', 'text-generation')
    print('Model downloaded successfully!')

asyncio.run(download())
"
```

## Documentation

See [docs/](docs/) for detailed documentation:
- [Architecture](docs/architecture.md) - System design and components
- [API Reference](docs/api.md) - Model management operations
- [Development Guide](docs/development.md) - Setup and coding standards
- [Deployment Guide](docs/deployment.md) - Production deployment

## Integration

The Model Manager integrates with:
- **Cluster Manager**: Provides model files to worker nodes
- **Gateway Manager**: Shares model availability information
- **Task Manager**: Supports model-specific task routing

## Configuration

Key environment variables:
```bash
MODEL_STORAGE_DIR=./models
MODEL_REGISTRY_FILE=./models/model_registry.json
HF_TOKEN=your_huggingface_token_here
```

See [.env.example](.env.example) for complete configuration options.