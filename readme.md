# GPU Benchmark Suite: Baseline Neural Networks

Native GPU benchmarking suite for 15 baseline neural network models with raw measurement capture and comprehensive statistical analysis.

**Models**: D1–D5 (DNN), R1–R5 (RNN), C1–C5 (CNN)  
**Backend**: PyTorch 2.5.1 + CUDA 12.1  
**Measurement**: Individual sample timing with full statistical distribution (mean, std, min, max, percentiles)

---

## System Requirements

- **GPU**: NVIDIA GeForce RTX 3060 or equivalent (12GB VRAM)
- **CUDA**: CUDA Toolkit 12.1+
- **Python**: 3.10+
- **OS**: Windows 10+ / Linux

---

## Installation

### 1. Create Python Environment

```bash
# Using conda (recommended)
conda create -n tf-gpu python=3.10 -y
conda activate tf-gpu

# Or using venv
python -m venv tf-gpu
tf-gpu\Scripts\activate  # Windows
source tf-gpu/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install manually with PyTorch CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas pyyaml torchtext
```

### 3. Verify GPU Detection

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
PyTorch: 2.5.1+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
```

---

## Benchmarking

### Quick Start

```bash
# Run all 15 models (1 trial)
python benchmark_pytorch_cuda.py --trials 1

# Run specific family
python benchmark_pytorch_cuda.py --models D1 D2 D3 D4 D5 --trials 1

# Custom output filename
python benchmark_pytorch_cuda.py --trials 1 --output my_results.csv

# Multiple trials with custom inference iterations
python benchmark_pytorch_cuda.py --trials 3 --inference-iters 100
```

### Command Arguments

```
--trials N              Number of trials per model (default: 1)
--models MODEL [...]   Specific models to benchmark (default: all 15)
--inference-iters N    Inference measurements per model (default: 100)
--output PATH          Output CSV path (default: auto-generated timestamp)
--cooldown SECONDS     Cooldown between models (default: 2)
```

### Benchmark Phases

Each model goes through 7 phases:

1. **Phase 1**: Model Creation
2. **Phase 2**: Data Loading (CIFAR-10)
3. **Phase 3**: First Inference (cold-start)
4. **Phase 4**: Warmup (4 untimed iterations)
5. **Phase 5**: Training (10 epochs)
6. **Phase 6**: Inference Timing (100 individual samples with raw measurement capture)
7. **Phase 7**: Cleanup & Memory Release

### Output Format

Two CSV files are generated per run:

#### Summary Statistics (`benchmark_results_pytorch_cuda_TIMESTAMP.csv`)

Columns:
- `model`: Model identifier (D1-D5, R1-R5, C1-C5)
- `model_family`: Family (DNN, RNN, CNN)
- `trial_id`: Trial number
- `device`: GPU device name
- `first_call_ms`: Cold-start latency (ms)
- `training_time_ms`: Total training time (ms)
- `training_unit`: Training configuration string
- `inference_mean_ms`: Mean inference latency (ms)
- `inference_std_ms`: Standard deviation (ms)
- `inference_min_ms`: Minimum latency (ms)
- `inference_max_ms`: Maximum latency (ms)
- `inference_p50_ms`: 50th percentile latency (ms)
- `inference_p95_ms`: 95th percentile latency (ms)
- `inference_p99_ms`: 99th percentile latency (ms)
- `inference_iters`: Number of measurements
- `raw_times_count`: Count of raw measurements captured
- `backend`: Backend (PyTorch-CUDA)

#### Raw Measurements (`benchmark_results_pytorch_cuda_TIMESTAMP_raw_timings.csv`)

Raw individual latency measurements for statistical analysis:
- `model`: Model identifier
- `trial_id`: Trial number
- `measurement_idx`: Sequential measurement index (0-99)
- `time_ms`: Individual inference latency (ms)

### Example Output Analysis

**Model Performance Summary** (inference latency, ms):

| Model | Family | Mean | Std | Min | Max | P95 |
|-------|--------|------|-----|-----|-----|-----|
| D1 | DNN | 0.079 | 0.018 | 0.071 | 0.244 | 0.092 |
| R5 | RNN | 3.737 | 0.264 | 3.440 | 4.551 | 4.227 |
| C5 | CNN | 0.378 | 0.084 | 0.344 | 0.912 | 0.535 |

---

## Model Families

### DNN (Dense Neural Networks)
- **D1**: Simple baseline (2 hidden layers)
- **D2-D5**: Progressive complexity (4-7 hidden layers)
- **Typical latency**: 0.08-0.25 ms

### RNN (Recurrent Neural Networks)
- **R1**: Simple RNN (1 layer)
- **R2-R5**: Complex stacks (2-5 layers)
- **Typical latency**: 0.26-3.74 ms
- **Note**: Larger latencies due to temporal computation

### CNN (Convolutional Neural Networks)
- **C1**: Simple baseline (2 conv blocks)
- **C2-C5**: Complex architectures (3-5 conv blocks)
- **Typical latency**: 0.27-0.41 ms

---

## Repository Structure

```
WebGPU-Study-Baseline/
├── DNN/                           (D1-D5 implementations)
│   ├── D1.py, D1.js              
│   ├── D2.py, D2.js              
│   ├── D3.py, D3.js              
│   ├── D4.py, D4.js              
│   └── D5.py, D5.js              
├── RNN/                           (R1-R5 implementations)
│   ├── R1.py, R1.js              
│   ├── R2.py, R2.js              
│   ├── R3.py, R3.js              
│   ├── R4.py, R4.js              
│   └── R5.py, R5.js              
├── CNN/                           (C1-C5 implementations)
│   ├── C1.py, C1.js              
│   ├── C2.py, C2.js              
│   ├── C3.py, C3.js              
│   ├── C4.py, C4.js              
│   └── C5.py, C5.js              
├── data/                          (Data loading utilities)
│   ├── data_cifar10.py            (PyTorch CIFAR-10 loader)
│   ├── data_cifar10.js            (JavaScript loader)
│   └── data_mnist.js              (MNIST loader)
├── benchmark_pytorch_cuda.py      (Main GPU benchmark engine)
├── benchmark_numba_cuda.py        (Alternative Numba-based benchmark)
├── benchmark_colab.ipynb          (Jupyter notebook for Colab)
├── requirements.txt               (Python dependencies)
├── docker-compose.yml             (Docker configuration)
├── Dockerfile                     (Container image)
├── server.js                      (Node.js web server)
├── index.html                     (Web interface)
├── style.css                      (Web styling)
└── readme.md                      (This file)
```

---

## Benchmarking Scripts

### PyTorch CUDA (`benchmark_pytorch_cuda.py`)

**Recommended for maximum raw GPU speed**

- Native PyTorch implementation
- Direct CUDA integration
- cuDNN autotuner enabled
- Individual sample timing (no batching aggregation)
- Full statistical distribution capture

### Numba CUDA (`benchmark_numba_cuda.py`)

Alternative NumPy-based implementation with JIT compilation.

### Jupyter Notebook (`benchmark_colab.ipynb`)

Google Colab compatible notebook for cloud GPU benchmarking.

---

## Web Interface

Run trained models in browser:

```bash
# Install Node.js dependencies
npm install

# Start server
node server.js
```

Access at `http://localhost:5500` and select model in web interface.

---

## Troubleshooting

### CUDA Not Detected
```bash
# Verify PyTorch CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

### Out of Memory
- Reduce batch size in source code
- Use `--inference-iters 50` for fewer measurements
- Monitor with `nvidia-smi` during benchmark

### Slow Performance
- Verify GPU utilization: `nvidia-smi -l 1` (loop every second)
- Check system thermal throttling
- Ensure cuDNN optimization is enabled

---

## References

- [PyTorch Official](https://pytorch.org/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [cuDNN Documentation](https://developer.nvidia.com/cudnn)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)