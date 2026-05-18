FROM nvidia/cuda:12.4-runtime-ubuntu22.04 AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ai/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

FROM base AS runtime
COPY ai/ /app/ai/
COPY ml/ /app/ml/
COPY vector_memory/ /app/vector_memory/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TF_CPP_MIN_LOG_LEVEL=3

CMD ["python3", "-m", "ai.orchestrator"]
