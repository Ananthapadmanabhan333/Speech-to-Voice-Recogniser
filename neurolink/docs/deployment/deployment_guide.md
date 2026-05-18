# Deployment Guide

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 16+ cores |
| RAM | 8 GB | 32 GB |
| GPU | - | NVIDIA GPU (8GB+ VRAM) |
| Disk | 20 GB | 100 GB SSD |
| OS | Ubuntu 22.04 / macOS 14 / Windows 11 | Ubuntu 24.04 |

### Software Requirements

- Docker 24+ and Docker Compose v2
- Python 3.12+
- Node.js 22+
- PostgreSQL 16+
- Redis 7+
- NVIDIA Container Toolkit (for GPU support)
- Make (optional, for convenience commands)

### Account Setup

- GitHub account (for CI/CD)
- Container registry (GitHub Container Registry, Docker Hub, or ECR)
- Cloud provider account (optional, for cloud deployment)

---

## Local Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/neurolink.git
cd neurolink
```

### 2. Environment Configuration

```bash
cp .env.example .env
```

Edit `.env` with your local configuration:

```env
# App
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# Database
DATABASE_URL=postgresql+asyncpg://neurolink:neurolink@localhost:5432/neurolink

# Redis
REDIS_URL=redis://localhost:6379

# JWT
JWT_SECRET=your-local-development-secret

# Feature flags
ENABLE_GESTURE_RECOGNITION=true
ENABLE_SPEECH_PROCESSING=true
ENABLE_EMOTION_DETECTION=true
ENABLE_TRANSLATION=true
ENABLE_PERSONALIZATION=true
```

### 3. Start Infrastructure Services

```bash
# Start PostgreSQL and Redis
docker compose -f docker/docker-compose.yml up -d postgres redis chromadb

# Verify services
docker compose -f docker/docker-compose.yml ps
```

### 4. Backend Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r backend/requirements.txt
pip install -r ai/requirements.txt

# Run database migrations
cd backend && alembic upgrade head && cd ..

# Seed demo data
npm run db:seed

# Start backend
npm run backend:dev
```

### 5. Frontend Setup

```bash
# Install Node dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000`.

### 6. Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# Should return:
# {"status":"healthy","version":"1.0.0","environment":"development",...}
```

---

## Docker Deployment

### Build Images

```bash
# Build all services
npm run docker:build

# Or build individually
docker compose -f docker/docker-compose.yml build backend
docker compose -f docker/docker-compose.yml build frontend
docker compose -f docker/docker-compose.yml build ai-worker
```

### Run with Docker Compose

```bash
# Start all services
npm run docker:up

# Or with custom environment
ENVIRONMENT=production docker compose -f docker/docker-compose.yml up -d
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| Backend | 8000 | FastAPI REST + WebSocket |
| Frontend | 3000 | Next.js web application |
| AI Worker | 9091 | ML inference worker |
| PostgreSQL | 5432 | Primary database |
| Redis | 6379 | Cache + pub/sub |
| ChromaDB | 8001 | Vector database |
| NGINX | 80/443 | Reverse proxy |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3001 | Dashboards |

### Docker Compose Profiles

```bash
# Full stack with monitoring
docker compose --profile monitoring up -d

# Production stack without dev tools
docker compose --profile production up -d

# Edge device stack (lightweight)
docker compose --profile edge up -d
```

---

## Edge Deployment

### NVIDIA Jetson (Orin/Nano)

#### Prerequisites

- Jetson device with JetPack 6.0+
- NVIDIA Container Toolkit for JetPack
- 8GB+ free disk space

#### Installation

```bash
# On the Jetson device
git clone https://github.com/your-org/neurolink.git
cd neurolink

# Install system dependencies
sudo apt update
sudo apt install -y python3-pip python3-dev libopenblas-dev

# Set up Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r edge/requirements.txt

# Install TensorRT (if not pre-installed)
pip install tensorrt==10.5.0

# Run edge deployment script
python edge/deployment/deploy_jetson.py --device jetson-orin --gpu-memory 16

# Start edge agent
python edge/deployment/edge_agent.py --config edge/jetson/config.yaml
```

#### Optimization

```bash
# Convert PyTorch models to TensorRT
python edge/optimization/export_tensorrt.py \
  --model-dir /path/to/models \
  --output-dir /opt/neurolink/engine \
  --precision fp16

# Benchmark
python edge/optimization/benchmark.py \
  --engine /opt/neurolink/engine/gesture.plan \
  --iterations 1000
```

### Raspberry Pi 5

#### Prerequisites

- Raspberry Pi 5 (8GB recommended)
- Raspberry Pi OS Bookworm (64-bit)
- 4GB+ free disk space

#### Installation

```bash
# On the Raspberry Pi
git clone https://github.com/your-org/neurolink.git
cd neurolink

# Install system dependencies
sudo apt update
sudo apt install -y python3-pip python3-dev libatlas-base-dev

# Set up Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://www.piwheels.org/simple -r edge/requirements.txt

# Install ONNX Runtime
pip install onnxruntime==1.20.0

# Run edge deployment script
python edge/deployment/deploy_rpi.py --device rpi5

# Start edge agent
python edge/deployment/edge_agent.py --config edge/raspberry_pi/config.yaml
```

#### Optimization

```bash
# Quantize models to INT8
python edge/optimization/quantize.py \
  --model-dir /path/to/models \
  --output-dir /opt/neurolink/engine \
  --precision int8 \
  --calibration-data /path/to/calibration

# Set performance mode
sudo cpufreq-set -g performance
```

---

## Cloud Deployment

### AWS Deployment

#### Architecture

```
Route 53 -> CloudFront -> ALB -> ECS Fargate (backend + frontend)
                                    -> ECS with GPU (ai-worker)
                                    -> RDS PostgreSQL
                                    -> ElastiCache Redis
                                    -> S3 (models + assets)
```

#### Terraform Deployment

```bash
# Navigate to infrastructure directory
cd infrastructure/aws

# Initialize Terraform
terraform init

# Review plan
terraform plan -var-file=environments/production.tfvars

# Apply
terraform apply -var-file=environments/production.tfvars
```

#### Manual ECS Deployment

```bash
# Authenticate with ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push images
docker tag neurolink-backend:latest <account>.dkr.ecr.us-east-1.amazonaws.com/neurolink-backend:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/neurolink-backend:latest

# Update ECS service
aws ecs update-service --cluster neurolink --service backend --force-new-deployment
```

### GCP Deployment

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/<project>/neurolink-backend

# Deploy to Cloud Run
gcloud run deploy neurolink-backend \
  --image gcr.io/<project>/neurolink-backend \
  --platform managed \
  --memory 4Gi \
  --cpu 4 \
  --concurrency 80 \
  --set-env-vars "DATABASE_URL=...,REDIS_URL=..."

# Deploy AI worker to GKE
gcloud container clusters get-credentials neurolink-cluster
kubectl apply -f k8s/ai-worker-deployment.yaml
```

### Azure Deployment

```bash
# Login to Azure Container Registry
az acr login --name neurolink

# Tag and push
docker tag neurolink-backend neurolink.azurecr.io/backend:latest
docker push neurolink.azurecr.io/backend:latest

# Deploy to Azure Container Instances
az container create \
  --resource-group neurolink \
  --name neurolink-backend \
  --image neurolink.azurecr.io/backend:latest \
  --cpu 4 --memory 8 \
  --environment-variables DATABASE_URL=... REDIS_URL=...

# Deploy to AKS
az aks get-credentials --resource-group neurolink --name neurolink-aks
kubectl apply -f k8s/
```

---

## Environment Variables

### Backend

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ENVIRONMENT` | No | `development` | Runtime environment (`development`, `staging`, `production`) |
| `DATABASE_URL` | Yes | - | PostgreSQL connection string |
| `REDIS_URL` | Yes | - | Redis connection string |
| `JWT_SECRET` | Yes | - | JWT signing secret (min 32 chars in production) |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |
| `CORS_ORIGINS` | No | `*` | Allowed CORS origins |
| `ENABLE_GESTURE_RECOGNITION` | No | `true` | Feature flag |
| `ENABLE_SPEECH_PROCESSING` | No | `true` | Feature flag |
| `ENABLE_EMOTION_DETECTION` | No | `true` | Feature flag |
| `ENABLE_TRANSLATION` | No | `true` | Feature flag |
| `ENABLE_PERSONALIZATION` | No | `true` | Feature flag |
| `WHISPER_MODEL_SIZE` | No | `base` | Whisper model size (`base`, `small`, `medium`, `large`) |
| `BCRYPT_ROUNDS` | No | `12` | Password hashing rounds |
| `OTLP_ENDPOINT` | No | `http://localhost:4318` | OpenTelemetry collector endpoint |

### Frontend

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | Yes | - | Backend API URL |
| `NEXT_PUBLIC_WS_URL` | Yes | - | WebSocket URL |
| `NEXT_PUBLIC_SENTRY_DSN` | No | - | Sentry error tracking DSN |

### AI Worker

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REDIS_URL` | Yes | - | Redis connection string |
| `POSTGRES_URL` | Yes | - | PostgreSQL connection string |
| `MODEL_CACHE` | No | `/data/models` | Model cache directory |
| `CHROMA_DB_PATH` | No | `/data/chroma` | ChromaDB persistence path |
| `ENABLE_GPU_INFERENCE` | No | `true` | GPU inference flag |
| `MODEL_PREFERENCE` | No | `balanced` | `speed`, `balanced`, `accuracy` |

---

## Troubleshooting

### Common Issues

#### Database Connection Failed

```bash
# Verify PostgreSQL is running
docker compose ps | grep postgres

# Check connection
psql -h localhost -U neurolink -d neurolink -c "SELECT 1"

# Verify DATABASE_URL format:
# postgresql+asyncpg://user:password@host:port/database
```

#### GPU Not Detected

```bash
# Check NVIDIA drivers
nvidia-smi

# Verify NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Ensure GPU reservation in docker-compose.yml
```

#### Model Loading Errors

```bash
# Check model cache
ls -la /data/models/

# Clear and re-download
rm -rf /data/models/*
docker compose restart ai-worker

# Verify model compatibility
python -c "import torch; print(torch.__version__)"
```

#### WebSocket Connection Failures

```bash
# Check WebSocket endpoint
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" -H "Host: localhost:8000" http://localhost:8000/ws

# Verify JWT token for WebSocket
# WebSocket requires a valid JWT token as query param: ws://host:8000/ws?token=<jwt>
```

#### High Memory Usage

```bash
# Check memory per service
docker stats

# Adjust model concurrency
# Set MAX_CONCURRENT_INFERENCE in AI worker env
# Reduce model cache size in settings

# For edge devices, use INT8 quantization
```

#### Migration Failures

```bash
# Check migration history
docker compose run --rm backend alembic history

# View current revision
docker compose run --rm backend alembic current

# Manual migration
docker compose run --rm backend alembic upgrade head

# Rollback one version
docker compose run --rm backend alembic downgrade -1

# Reset (destructive!)
docker compose run --rm backend alembic downgrade base
docker compose run --rm backend alembic upgrade head
```

### Logs

```bash
# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f backend
docker compose logs -f ai-worker

# View recent errors
docker compose logs --tail=100 backend | grep -i error

# Export logs
docker compose logs -t > neurolink-logs-$(date +%Y%m%d).txt
```

### Diagnostic Commands

```bash
# Full system health check
python cicd/scripts/health_check.py --base-url http://localhost:8000 --check-all

# Database health
docker compose exec postgres pg_isready -U neurolink

# Redis health
docker compose exec redis redis-cli ping

# ChromaDB health
curl http://localhost:8001/api/v1/version

# Backend metrics
curl http://localhost:8000/metrics

# Test inference
curl -X POST http://localhost:8000/api/v1/communication/translate \
  -H "Content-Type: application/json" \
  -d '{"source_text":"Hello","source_lang":"en","target_lang":"es"}'
```

### Getting Help

- GitHub Issues: [https://github.com/your-org/neurolink/issues](https://github.com/your-org/neurolink/issues)
- Documentation: [https://docs.neurolink.dev](https://docs.neurolink.dev)
- Discord: [https://discord.gg/neurolink](https://discord.gg/neurolink)
- Email: support@neurolink.dev
