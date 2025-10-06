FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04

WORKDIR /app

# Install Python and build tools
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    build-essential \
    libgl1 \
    git \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    poppler-utils \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    # Audio processing dependencies
    sox \
    libsox-dev \
    libsox-fmt-all \
    libsndfile1 \
    libsndfile-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && git config --global core.symlinks true

# Set python3 as default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install torch/torchaudio with CUDA 12.1 builds
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies in the correct order
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    numpy==1.23.5 \
    scipy \
    librosa \
    soundfile \
    numba \
    inflect \
    sentencepiece \
    sacrebleu \
    'huggingface-hub==0.17.3' \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0+cu121 \
    torchaudio==2.4.0+cu121 \
    && pip install --no-cache-dir \
    hydra-core \
    omegaconf \
    'transformers==4.33.0' \
    && pip install --no-cache-dir \
    'nemo_toolkit[asr]==1.21.0'

# Install NeMo and its dependencies
RUN pip install --no-cache-dir nemo_toolkit[asr]==1.20.0 && \
    pip install --no-cache-dir \
    hydra-core \
    omegaconf \
    soundfile \
    sox \
    unidecode \
    webdataset \
    youtokentome

# Copy requirements and install rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

EXPOSE 8288

CMD ["python", "-m", "app.main"]
