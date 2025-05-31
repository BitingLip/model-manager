# Model Manager Deployment Guide

## Production Deployment

### Deployment Architecture

```mermaid
graph TB
    subgraph "Production Environment"
        subgraph "Model Manager Cluster"
            MM1[Model Manager 1]
            MM2[Model Manager 2] 
            MM3[Model Manager 3]
        end
        
        subgraph "Storage Layer"
            NFS[Shared NFS Storage]
            S3[Object Storage S3]
            DB[(Registry Database)]
        end
        
        subgraph "Load Balancer"
            LB[Nginx/HAProxy]
        end
        
        subgraph "Monitoring"
            PROM[Prometheus]
            GRAF[Grafana]
            LOG[Centralized Logging]
        end
    end
    
    subgraph "External Services"
        HF[HuggingFace Hub]
        CDN[Content Delivery Network]
    end
    
    LB --> MM1
    LB --> MM2
    LB --> MM3
    
    MM1 --> NFS
    MM2 --> NFS
    MM3 --> NFS
    
    MM1 --> S3
    MM2 --> S3
    MM3 --> S3
    
    MM1 --> DB
    MM2 --> DB
    MM3 --> DB
    
    MM1 --> HF
    MM2 --> HF
    MM3 --> HF
    
    PROM --> MM1
    PROM --> MM2
    PROM --> MM3
    
    style MM1 fill:#e1f5fe
    style MM2 fill:#e1f5fe
    style MM3 fill:#e1f5fe
    style NFS fill:#e8f5e8
    style S3 fill:#fff3e0
    style DB fill:#f3e5f5
```

### Container Deployment

#### Dockerfile
```dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MODEL_STORAGE_DIR=/opt/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY app/ ./app/
COPY .env.production .env

# Create models directory and set permissions
RUN mkdir -p /opt/models && \
    chown -R app:app /app /opt/models

# Switch to app user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import app.models.registry; print('healthy')"

# Expose port (if running as API service)
EXPOSE 8005

# Start command
CMD ["python", "-m", "app.main"]
```

#### docker-compose.yml
```yaml
version: '3.8'

services:
  model-manager:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8005:8005"
    environment:
      - MODEL_STORAGE_DIR=/opt/models
      - MODEL_REGISTRY_FILE=/opt/models/registry.json
      - DATABASE_URL=postgresql://user:pass@db:5432/models
      - REDIS_URL=redis://redis:6379/0
      - HF_TOKEN=${HF_TOKEN}
    volumes:
      - model_storage:/opt/models
      - ./logs:/app/logs
    depends_on:
      - db
      - redis
    restart: unless-stopped
    networks:
      - model_network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8005/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=models
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    networks:
      - model_network

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    networks:
      - model_network
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - model-manager
    restart: unless-stopped
    networks:
      - model_network

volumes:
  model_storage:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/models
  postgres_data:
  redis_data:

networks:
  model_network:
    driver: bridge
```

### Kubernetes Deployment

#### namespace.yaml
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: model-manager
  labels:
    name: model-manager
```

#### configmap.yaml
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-manager-config
  namespace: model-manager
data:
  MODEL_STORAGE_DIR: "/opt/models"
  MODEL_REGISTRY_FILE: "/opt/models/registry.json"
  LOG_LEVEL: "INFO"
  MAX_CONCURRENT_DOWNLOADS: "3"
  DOWNLOAD_TIMEOUT: "3600"
```

#### secret.yaml
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: model-manager-secrets
  namespace: model-manager
type: Opaque
data:
  HF_TOKEN: <base64-encoded-token>
  DATABASE_URL: <base64-encoded-db-url>
  REDIS_URL: <base64-encoded-redis-url>
```

#### persistent-volume.yaml
```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-storage-pv
spec:
  capacity:
    storage: 1Ti
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: nfs
  nfs:
    server: nfs-server.example.com
    path: /data/models

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage-pvc
  namespace: model-manager
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 1Ti
  storageClassName: nfs
```

#### deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-manager
  namespace: model-manager
  labels:
    app: model-manager
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-manager
  template:
    metadata:
      labels:
        app: model-manager
    spec:
      containers:
      - name: model-manager
        image: model-manager:latest
        ports:
        - containerPort: 8005
        env:
        - name: MODEL_STORAGE_DIR
          valueFrom:
            configMapKeyRef:
              name: model-manager-config
              key: MODEL_STORAGE_DIR
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: model-manager-secrets
              key: HF_TOKEN
        volumeMounts:
        - name: model-storage
          mountPath: /opt/models
        - name: logs
          mountPath: /app/logs
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8005
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8005
          initialDelaySeconds: 10
          periodSeconds: 5
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc
      - name: logs
        emptyDir: {}
```

#### service.yaml
```yaml
apiVersion: v1
kind: Service
metadata:
  name: model-manager-service
  namespace: model-manager
spec:
  selector:
    app: model-manager
  ports:
    - protocol: TCP
      port: 8005
      targetPort: 8005
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-manager-ingress
  namespace: model-manager
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - models.example.com
    secretName: model-manager-tls
  rules:
  - host: models.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: model-manager-service
            port:
              number: 8005
```

## Environment Configuration

### Production Environment Variables

```bash
# Storage Configuration
MODEL_STORAGE_DIR=/opt/models
MODEL_REGISTRY_FILE=/opt/models/registry.json
BACKUP_STORAGE_DIR=/backup/models

# Database Configuration
DATABASE_URL=postgresql://user:password@db-host:5432/models
REDIS_URL=redis://redis-host:6379/0

# HuggingFace Configuration
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_HUB_CACHE=/opt/cache/huggingface

# Security Configuration
SECRET_KEY=your-secret-key-here
API_KEY_HEADER=X-API-Key
ALLOWED_HOSTS=models.example.com,api.example.com

# Performance Configuration
MAX_CONCURRENT_DOWNLOADS=5
DOWNLOAD_TIMEOUT=7200
WORKER_PROCESSES=4
WORKER_CONNECTIONS=1000

# Monitoring Configuration
PROMETHEUS_PORT=9090
LOG_LEVEL=INFO
LOG_FORMAT=json
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project

# SSL Configuration
SSL_CERT_PATH=/etc/ssl/certs/model-manager.crt
SSL_KEY_PATH=/etc/ssl/private/model-manager.key
```

### Configuration Management

#### Helm Chart Values (`values.yaml`)
```yaml
replicaCount: 3

image:
  repository: model-manager
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8005

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: models.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: model-manager-tls
      hosts:
        - models.example.com

storage:
  size: 1Ti
  storageClass: nfs
  accessMode: ReadWriteMany

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s

config:
  modelStorageDir: /opt/models
  logLevel: INFO
  maxConcurrentDownloads: 5
  downloadTimeout: 7200

secrets:
  hfToken: ""
  databaseUrl: ""
  redisUrl: ""
```

## Monitoring and Observability

### Prometheus Metrics

#### Custom Metrics
```python
# app/metrics.py
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

# Create custom registry
registry = CollectorRegistry()

# Define metrics
download_requests_total = Counter(
    'model_downloads_total',
    'Total number of model download requests',
    ['model_id', 'status'],
    registry=registry
)

download_duration_seconds = Histogram(
    'model_download_duration_seconds',
    'Time spent downloading models',
    ['model_id'],
    registry=registry
)

registry_size = Gauge(
    'model_registry_size',
    'Number of models in registry',
    registry=registry
)

storage_used_bytes = Gauge(
    'model_storage_used_bytes',
    'Storage space used by models',
    registry=registry
)

active_downloads = Gauge(
    'model_active_downloads',
    'Number of currently active downloads',
    registry=registry
)
```

#### Metrics Integration
```python
# app/models/downloader.py
import time
from app.metrics import download_requests_total, download_duration_seconds

async def download_model_hf(self, model_id: str, model_type: ModelType):
    start_time = time.time()
    
    try:
        # Download logic here
        result = await self._download_implementation(model_id, model_type)
        
        # Record success
        download_requests_total.labels(
            model_id=model_id, 
            status='success'
        ).inc()
        
        return result
        
    except Exception as e:
        # Record failure
        download_requests_total.labels(
            model_id=model_id, 
            status='error'
        ).inc()
        raise
        
    finally:
        # Record duration
        duration = time.time() - start_time
        download_duration_seconds.labels(model_id=model_id).observe(duration)
```

### Health Checks

#### Application Health Check
```python
# app/health.py
from typing import Dict, Any
from pathlib import Path
import psutil
import asyncio

class HealthChecker:
    def __init__(self, registry: ModelRegistry, storage_dir: Path):
        self.registry = registry
        self.storage_dir = storage_dir
    
    async def check_health(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        checks = {
            "registry": await self._check_registry(),
            "storage": await self._check_storage(),
            "memory": await self._check_memory(),
            "disk": await self._check_disk(),
            "downloads": await self._check_active_downloads()
        }
        
        overall_status = "healthy" if all(
            check["status"] == "healthy" for check in checks.values()
        ) else "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks
        }
    
    async def _check_registry(self) -> Dict[str, Any]:
        """Check registry accessibility."""
        try:
            models = self.registry.list_models()
            return {
                "status": "healthy",
                "models_count": len(models),
                "registry_file_exists": self.registry.registry_file.exists()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_storage(self) -> Dict[str, Any]:
        """Check storage accessibility and space."""
        try:
            if not self.storage_dir.exists():
                return {
                    "status": "unhealthy",
                    "error": "Storage directory does not exist"
                }
            
            # Check write permissions
            test_file = self.storage_dir / ".health_check"
            test_file.touch()
            test_file.unlink()
            
            # Get disk usage
            usage = psutil.disk_usage(str(self.storage_dir))
            free_percent = (usage.free / usage.total) * 100
            
            status = "healthy" if free_percent > 10 else "warning"
            if free_percent < 5:
                status = "unhealthy"
            
            return {
                "status": status,
                "free_space_percent": round(free_percent, 2),
                "free_space_gb": round(usage.free / (1024**3), 2),
                "total_space_gb": round(usage.total / (1024**3), 2)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        memory = psutil.virtual_memory()
        
        status = "healthy"
        if memory.percent > 90:
            status = "unhealthy"
        elif memory.percent > 80:
            status = "warning"
        
        return {
            "status": status,
            "memory_percent": memory.percent,
            "available_gb": round(memory.available / (1024**3), 2)
        }
```

### Logging Configuration

#### Structured Logging
```python
# app/logging.py
import structlog
import logging.config

def setup_production_logging():
    """Configure production logging with structured output."""
    
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.dev.JSONRenderer(),
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": "/app/logs/model-manager.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": True,
            },
        },
    }
    
    logging.config.dictConfig(LOGGING_CONFIG)
    
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
```

## Security Considerations

### API Security

#### Authentication and Authorization
```python
# app/security.py
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta

security = HTTPBearer()

class SecurityManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def create_access_token(self, data: dict, expires_delta: timedelta = None):
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm="HS256")
        return encoded_jwt
    
    def verify_token(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Verify JWT token."""
        try:
            payload = jwt.decode(
                credentials.credentials, 
                self.secret_key, 
                algorithms=["HS256"]
            )
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return username
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

# Usage in API endpoints
@app.get("/models")
async def list_models(current_user: str = Depends(security_manager.verify_token)):
    """List models - requires authentication."""
    return registry.list_models()
```

### File System Security

#### Secure File Operations
```python
# app/security/file_operations.py
import os
import stat
from pathlib import Path
from typing import Optional

class SecureFileOperations:
    @staticmethod
    def secure_path_join(base: Path, *paths: str) -> Path:
        """Securely join paths preventing directory traversal."""
        result = base
        for path in paths:
            # Remove any directory traversal attempts
            clean_path = os.path.normpath(path).replace('..', '')
            result = result / clean_path
        
        # Ensure result is still within base directory
        try:
            result.resolve().relative_to(base.resolve())
            return result
        except ValueError:
            raise ValueError("Path traversal attempt detected")
    
    @staticmethod
    def set_secure_permissions(file_path: Path, is_directory: bool = False):
        """Set secure file permissions."""
        if is_directory:
            # Directory: owner rwx, group rx, other none
            os.chmod(file_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
        else:
            # File: owner rw, group r, other none
            os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)
    
    @staticmethod
    def validate_file_type(file_path: Path, allowed_extensions: set) -> bool:
        """Validate file type by extension."""
        return file_path.suffix.lower() in allowed_extensions
```

## Backup and Recovery

### Automated Backup Strategy

#### Backup Script
```bash
#!/bin/bash
# backup-models.sh

set -e

# Configuration
BACKUP_DIR="/backup/models"
SOURCE_DIR="/opt/models"
RETENTION_DAYS=30
S3_BUCKET="model-backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create backup directory
mkdir -p "${BACKUP_DIR}/${TIMESTAMP}"

# Backup registry file
cp "${SOURCE_DIR}/registry.json" "${BACKUP_DIR}/${TIMESTAMP}/"

# Create compressed archive of models
echo "Creating backup archive..."
tar -czf "${BACKUP_DIR}/${TIMESTAMP}/models.tar.gz" \
    -C "${SOURCE_DIR}" \
    --exclude="cache/*" \
    --exclude="*.tmp" \
    .

# Upload to S3 (if configured)
if [ -n "$S3_BUCKET" ]; then
    echo "Uploading to S3..."
    aws s3 cp "${BACKUP_DIR}/${TIMESTAMP}/" \
        "s3://${S3_BUCKET}/$(date +%Y)/$(date +%m)/" \
        --recursive
fi

# Cleanup old backups
echo "Cleaning up old backups..."
find "${BACKUP_DIR}" -type d -mtime +${RETENTION_DAYS} -exec rm -rf {} +

echo "Backup completed: ${BACKUP_DIR}/${TIMESTAMP}"
```

#### Recovery Procedures
```bash
#!/bin/bash
# restore-models.sh

set -e

BACKUP_TIMESTAMP="$1"
BACKUP_DIR="/backup/models"
TARGET_DIR="/opt/models"

if [ -z "$BACKUP_TIMESTAMP" ]; then
    echo "Usage: $0 <backup_timestamp>"
    echo "Available backups:"
    ls -1 "$BACKUP_DIR"
    exit 1
fi

# Stop model manager service
systemctl stop model-manager

# Backup current state
mv "$TARGET_DIR" "${TARGET_DIR}.backup.$(date +%s)"

# Restore from backup
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# Extract backup
tar -xzf "${BACKUP_DIR}/${BACKUP_TIMESTAMP}/models.tar.gz"

# Restore registry
cp "${BACKUP_DIR}/${BACKUP_TIMESTAMP}/registry.json" "$TARGET_DIR/"

# Set permissions
chown -R app:app "$TARGET_DIR"
chmod -R 755 "$TARGET_DIR"

# Start service
systemctl start model-manager

echo "Restore completed from backup: $BACKUP_TIMESTAMP"
```

## Troubleshooting

### Common Issues

#### High Memory Usage
```python
# Monitor memory usage during downloads
import psutil
import gc

def monitor_memory_usage():
    """Monitor and log memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    logger.info(
        "Memory usage",
        rss_mb=memory_info.rss / 1024 / 1024,
        vms_mb=memory_info.vms / 1024 / 1024,
        percent=process.memory_percent()
    )
    
    # Force garbage collection if memory usage is high
    if process.memory_percent() > 80:
        gc.collect()
        logger.warning("High memory usage detected, forced garbage collection")
```

#### Storage Issues
```python
# Check and clean up storage
async def cleanup_storage():
    """Clean up old and orphaned files."""
    
    # Remove incomplete downloads
    for temp_file in storage_dir.glob("*.tmp"):
        if temp_file.stat().st_mtime < time.time() - 3600:  # 1 hour old
            temp_file.unlink()
            logger.info("Removed stale temp file", file=str(temp_file))
    
    # Remove orphaned model directories
    registry_models = {entry.model_id for entry in registry.list_models()}
    
    for model_dir in storage_dir.iterdir():
        if model_dir.is_dir() and model_dir.name not in registry_models:
            logger.warning("Found orphaned model directory", dir=str(model_dir))
            # Optionally remove or move to quarantine
```

#### Network Issues
```python
# Implement retry logic with exponential backoff
import asyncio
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def download_with_retry(url: str, local_path: Path):
    """Download with retry logic."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
```

### Performance Tuning

#### Database Optimization
```sql
-- PostgreSQL optimization for model registry
CREATE INDEX idx_models_status ON models(status);
CREATE INDEX idx_models_type ON models(model_type);
CREATE INDEX idx_models_download_date ON models(download_date);

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM models WHERE status = 'available';
```

#### Caching Strategy
```python
# Implement Redis caching for frequently accessed data
import redis
import json
from typing import Optional

class ModelCache:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
    
    async def get_model(self, model_id: str) -> Optional[dict]:
        """Get model from cache."""
        cached = self.redis.get(f"model:{model_id}")
        if cached:
            return json.loads(cached)
        return None
    
    async def set_model(self, model_id: str, model_data: dict, ttl: int = 3600):
        """Cache model data."""
        self.redis.setex(
            f"model:{model_id}",
            ttl,
            json.dumps(model_data, default=str)
        )
    
    async def invalidate_model(self, model_id: str):
        """Remove model from cache."""
        self.redis.delete(f"model:{model_id}")
```
