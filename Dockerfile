# CPU image for training and evaluating the subfield classifier.
# The model is a small MLP (about 335k parameters) on precomputed embeddings, so no GPU is needed.
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends git curl jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install the pinned dependencies first so that this layer is cached
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["bash"]
