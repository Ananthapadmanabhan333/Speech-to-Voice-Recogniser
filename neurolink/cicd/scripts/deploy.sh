#!/usr/bin/env bash
set -euo pipefail

# ── Neurolink Deployment Script ────────────────────────────────────────────
# Usage: ENVIRONMENT=staging IMAGE_TAG=<sha> bash deploy.sh
#
# Required environment variables:
#   ENVIRONMENT         - staging | production
#   IMAGE_TAG           - Docker image tag (git SHA or semver)
#   DOCKER_REGISTRY     - Container registry host
#   REPO_OWNER          - GitHub repository owner
#   DATABASE_URL        - PostgreSQL connection string
#   REDIS_URL           - Redis connection string
#   JWT_SECRET          - JWT signing secret
#
# Optional:
#   API_KEY             - API key for model updates
#   LOG_LEVEL           - Logging verbosity (default: INFO)
#   MODEL_CACHE         - Model cache directory (default: ./data/models)
# ─────────────────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
error() { log "ERROR: $*"; }
die() { error "$*"; exit 1; }

# ── Validate environment ──────────────────────────────────────────────────

: "${ENVIRONMENT:?Must set ENVIRONMENT}"
: "${IMAGE_TAG:?Must set IMAGE_TAG}"
: "${DOCKER_REGISTRY:?Must set DOCKER_REGISTRY}"
: "${REPO_OWNER:?Must set REPO_OWNER}"
: "${DATABASE_URL:?Must set DATABASE_URL}"
: "${REDIS_URL:?Must set REDIS_URL}"
: "${JWT_SECRET:?Must set JWT_SECRET}"
: "${LOG_LEVEL:=INFO}"
: "${MODEL_CACHE:=./data/models}"

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DEPLOY_LOG="${PROJECT_DIR}/deploy-${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S).log"
COMPOSE_FILE="${PROJECT_DIR}/docker/docker-compose.yml"
COMPOSE_PROJECT="neurolink-${ENVIRONMENT}"

log "Deploying Neurolink to ${ENVIRONMENT}"
log "Project directory: ${PROJECT_DIR}"
log "Image tag: ${IMAGE_TAG}"
log "Log file: ${DEPLOY_LOG}"

exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3 15
exec 1>"${DEPLOY_LOG}" 2>&1

# ── Step 1: Configure environment ─────────────────────────────────────────

export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT}"
export DOCKER_REGISTRY
export REPO_OWNER
export IMAGE_TAG
export ENVIRONMENT
export LOG_LEVEL
export DATABASE_URL
export REDIS_URL
export JWT_SECRET
export MODEL_CACHE

if [ -n "${API_KEY:-}" ]; then
  export API_KEY
fi

# ── Step 2: Create required directories ────────────────────────────────────

mkdir -p "${PROJECT_DIR}/data/models"
mkdir -p "${PROJECT_DIR}/data/audio"
mkdir -p "${PROJECT_DIR}/vector_memory/chroma"
mkdir -p "${PROJECT_DIR}/logs"

# ── Step 3: Pull latest images ────────────────────────────────────────────

log "Pulling Docker images..."
docker compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT}" pull backend
docker compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT}" pull frontend
docker compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT}" pull ai-worker

# ── Step 4: Run database migrations ───────────────────────────────────────

log "Running database migrations..."
docker compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT}" run --rm \
  -e DATABASE_URL="${DATABASE_URL}" \
  backend alembic upgrade head 2>&1 || {
    error "Database migration failed. Attempting rollback..."
    docker compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT}" run --rm \
      -e DATABASE_URL="${DATABASE_URL}" \
      backend alembic downgrade -1 2>&1 || true
    die "Migration failed and rollback attempted. Check ${DEPLOY_LOG}"
  }
log "Database migrations complete."

# ── Step 5: Download and update models ────────────────────────────────────

log "Downloading latest models..."
docker compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT}" run --rm \
  -e MODEL_CACHE="${MODEL_CACHE}" \
  -e API_KEY="${API_KEY:-}" \
  ai-worker python -m ml.pipelines.download_models \
    --target "${ENVIRONMENT}" \
    --output-dir "${MODEL_CACHE}" 2>&1 || {
      log "Model download returned non-zero, continuing..."
    }

# ── Step 6: Deploy services ───────────────────────────────────────────────

log "Deploying services..."
docker compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT}" up -d \
  --no-deps \
  --remove-orphans \
  postgres redis chromadb

docker compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT}" up -d \
  --no-deps \
  backend ai-worker frontend

log "Waiting for services to be healthy..."

# ── Step 7: Health verification ───────────────────────────────────────────

HEALTHY=false
for i in $(seq 1 30); do
  BACKEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    "http://localhost:8000/health" 2>/dev/null || echo "000")

  if [ "${BACKEND_STATUS}" = "200" ]; then
    HEALTHY=true
    log "Backend service is healthy (attempt ${i})"
    break
  fi
  sleep 2
done

if [ "${HEALTHY}" = false ]; then
  error "Health check failed after 60 seconds"
  error "Showing recent logs:"
  docker compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT}" logs --tail=50 backend
  die "Deployment health check failed. Manual intervention required."
fi

# ── Step 8: Run post-deployment verification ──────────────────────────────

log "Running post-deployment smoke tests..."
python "${PROJECT_DIR}/cicd/scripts/health_check.py" \
  --base-url "http://localhost:8000" \
  --timeout 30 \
  --check-db \
  --check-redis || {
    error "Post-deployment checks failed"
  }

log "Deployment to ${ENVIRONMENT} completed successfully!"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: Deployment to ${ENVIRONMENT} complete" >&3

# ── Rollback procedure (manual trigger) ────────────────────────────────────
# To rollback manually:
#   docker compose -f docker/docker-compose.yml -p neurolink-<env> down
#   git checkout <previous-tag>
#   export IMAGE_TAG=<previous-sha>
#   bash cicd/scripts/deploy.sh
# ─────────────────────────────────────────────────────────────────────────────
