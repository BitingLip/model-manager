#!/usr/bin/env python3
"""
Model Manager Startup Script
Handles proper path setup for the model manager application
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set working directory
os.chdir(project_root)

# Import and run the main application
if __name__ == "__main__":
    from app.main import app
    import uvicorn
    
    # Load settings
    from app.config import get_settings
    settings = get_settings()    # Start the server
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=False,  # Disable reload to avoid issues
        log_level="info"
    )
