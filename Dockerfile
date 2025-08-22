FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04

WORKDIR /app

# Install Python and build tools
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    build-essential \
    git \
    # Tesseract OCR and related libs
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install torch/torchaudio with CUDA 12.1 builds
RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    torch==2.4.0+cu121 \
    torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

EXPOSE 8288

CMD ["python", "-m", "app.main"]
