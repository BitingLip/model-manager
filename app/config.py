"""
Model Manager Configuration
Uses the new distributed configuration system for microservice independence.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path to access config package
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import from the config package (now available in __init__.py)
from config.distributed_config import load_service_config, load_infrastructure_config
from config.service_discovery import ServiceDiscovery


class ModelManagerSettings:
    """Model Manager specific configuration adapter using distributed config"""
    
    def __init__(self):
        # Load service-specific configuration
        self.config = load_service_config('model-manager', 'manager')
        
        # Load infrastructure configuration for shared resources
        self.infrastructure = load_infrastructure_config()
        
        # Initialize service discovery
        try:
            self.service_discovery = ServiceDiscovery()
        except Exception as e:
            print(f"Warning: Could not initialize service discovery: {e}")
            self.service_discovery = None
    
    def get_config_value(self, key: str, default: str = '') -> str:
        """Get configuration value with fallback to environment variables"""
        return self.config.get(key, os.getenv(key, default))
    
    @property
    def host(self):
        return self.get_config_value('MODEL_MANAGER_HOST', 'localhost')
    
    @property 
    def port(self):
        return int(self.get_config_value('MODEL_MANAGER_PORT', '8001'))
    
    @property
    def debug(self):
        return self.get_config_value('DEBUG', 'true').lower() == 'true'
    
    @property
    def cors_origins(self):
        origins = self.get_config_value('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173')
        return [origin.strip() for origin in origins.split(',')]
    
    @property
    def model_cache_dir(self):
        return self.get_config_value('MODEL_CACHE_DIR', './models')
    
    @property
    def max_model_cache_size_gb(self):
        return float(self.get_config_value('MAX_MODEL_CACHE_SIZE_GB', '50.0'))
    
    @property
    def model_load_timeout(self):
        return int(self.get_config_value('MODEL_LOAD_TIMEOUT', '300'))
    
    @property
    def db_host(self):
        return self.get_config_value('MODEL_DB_HOST', 'localhost')
    
    @property
    def db_port(self):
        return int(self.get_config_value('MODEL_DB_PORT', '5432'))
    
    @property
    def db_name(self):
        return self.get_config_value('MODEL_DB_NAME', 'bitinglip_models')
    
    @property
    def db_user(self):
        return self.get_config_value('MODEL_DB_USER', 'bitinglip')
    
    @property
    def db_password(self):
        return self.get_config_value('MODEL_DB_PASSWORD', 'secure_password')
    
    @property
    def huggingface_token(self):
        return self.get_config_value('HUGGINGFACE_TOKEN', '')
    
    @property
    def log_level(self):
        return self.get_config_value('LOG_LEVEL', 'INFO')


def get_settings():
    """Get model manager settings instance"""
    return ModelManagerSettings()

# Create default instance
settings = get_settings()

# Backward compatibility alias
Settings = ModelManagerSettings

# Export the same interface as before for backward compatibility
__all__ = [
    'Settings', 'get_settings', 'ModelManagerSettings', 'settings'
]
