"""
Model Manager Configuration
Uses centralized BitingLip configuration system.
"""

# Import from centralized configuration system
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../config'))

from central_config import get_config
from service_discovery import ServiceDiscovery

class ModelManagerSettings:
    """Model Manager specific configuration adapter"""
    
    def __init__(self):
        self.config = get_config('model_manager')
        self.service_discovery = ServiceDiscovery()
    
    @property
    def host(self):
        return self.config.model_manager_host
    
    @property 
    def port(self):
        return self.config.model_manager_port
    
    @property
    def debug(self):
        return self.config.debug
    
    @property
    def cors_origins(self):
        return self.config.cors_origins

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
