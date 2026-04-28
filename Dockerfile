# Dockerfile for GPU-accelerated benchmarking
# Uses NVIDIA CUDA base image with TensorFlow GPU support

FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Set working directory
WORKDIR /benchmark

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1

# Install Python 3.10 and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install TensorFlow GPU (will use CUDA 11.2 from base image)
RUN pip install --no-cache-dir tensorflow[and-cuda]==2.13.1

# Install additional dependencies for benchmarking
RUN pip install --no-cache-dir \
    numpy>=1.21 \
    pandas \
    matplotlib \
    scikit-learn

# Verify CUDA and cuDNN availability
RUN python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0); print('Built with CUDA:', tf.test.is_built_with_cuda())"

# Copy benchmark code into container
COPY benchmark_native.py .
COPY DNN ./DNN
COPY RNN ./RNN
COPY CNN ./CNN
COPY data ./data

# Create output directory for results
RUN mkdir -p /benchmark/results

# Default command: run benchmark for all models with 1 trial
CMD ["python3", "benchmark_native.py", "--trials", "1", "--output", "/benchmark/results/benchmark_results.csv"]
