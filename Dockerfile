# ==== BASE IMAGE (CUDA for GPU training) ====
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ==== SYSTEM SETUP ====
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git wget curl vim build-essential python3 python3-pip python3-venv jq \
    && rm -rf /var/lib/apt/lists/*

# Use python3 by default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN python -m pip install --upgrade pip

# ==== PYTHON DEPENDENCIES ====
# Install core ML dependencies
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Hugging Face + training ecosystem
RUN pip install \
    transformers \
    datasets \
    evaluate \
    accelerate \
    sentencepiece \
    sacrebleu \
    rouge-score \
    nltk \
    wandb \
    tqdm

# ==== NLTK DATA ====
RUN python -m nltk.downloader punkt

# ==== WORKDIR ====
WORKDIR /workspace

# ==== COPY PROJECT FILES ====
# (add only requirements first for caching)
COPY requirements.txt .
RUN pip install -r requirements.txt || true

# Now copy the rest of the repo
COPY . .

# ==== ENV VARS ====
# prevent huggingface token caching errors in containers
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/transformers
ENV WANDB_CACHE_DIR=/workspace/.cache/wandb

# Set CUDA device visibility inside container
ENV CUDA_VISIBLE_DEVICES=0

# ==== DEFAULT COMMAND ====
CMD ["bash"]
