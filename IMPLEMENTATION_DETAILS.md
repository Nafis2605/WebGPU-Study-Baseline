# GPU Benchmark Suite - Implementation Details

## Framework & Versions

**Primary Framework**: PyTorch 2.5.1+cu121  
**GPU Acceleration Library**: TorchVision 0.16.1  
**CUDA Toolkit**: 12.1  
**Python**: 3.10.7  
**Operating System**: Windows 10+

---

## Hardware Configuration

**GPU**: NVIDIA GeForce RTX 3060  
**VRAM**: 12.0 GB  
**Compute Capability**: 8.6  
**Driver Version**: 581.95 (or latest NVIDIA Game Ready Driver)

---

## Python Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.5.1+cu121 | Neural network framework with CUDA support |
| TorchVision | 0.16.1 | Computer vision utilities, CIFAR-10 loader |
| NumPy | 1.24.0+ | Numerical computations, statistics |
| Pandas | 1.5.0+ | Data analysis and CSV handling |

**Installation Command**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy>=1.24.0 pandas>=1.5.0
```

---

## Benchmark Protocol

### Measurement Methodology
- **Trials**: 1 (configurable via `--trials N`)
- **Inference Iterations**: 100 individual samples per model
- **Total Raw Measurements**: 1,500 (15 models × 100 samples)

### Timing Mechanism
- **Synchronization**: `torch.cuda.synchronize()` for GPU-CPU sync
- **Timer**: `time.perf_counter()` (nanosecond precision)
- **Precision**: Microsecond-level accuracy
- **Batch Mode**: Individual sample measurement (batch_size=1)

### cuDNN Optimization
- **Autotuner**: Enabled (`torch.backends.cudnn.benchmark = True`)
- **Effect**: Finds fastest cuDNN kernels per operation

### Benchmark Phases per Model
1. **Phase 1**: Model Creation (GPU memory allocation)
2. **Phase 2**: CIFAR-10 Data Loading (train 5000, test 1000 samples)
3. **Phase 3**: First Inference Cold-Start (single unsynced call)
4. **Phase 4**: Warmup (4 untimed iterations for GPU cache warming)
5. **Phase 5**: Training (10 epochs, batch_size varies by model family)
6. **Phase 6**: Inference Timing (100 individual sample measurements, fully synchronized)
7. **Phase 7**: Cleanup & Memory Release

---

## Model Architecture

### 15 Total Models (All Implemented in PyTorch)

#### DNN (Dense Neural Networks) - D1 to D5
- **D1**: 2 hidden layers (32, 16 units) - baseline
- **D2**: 4 hidden layers (64, 48, 32, 16 units)
- **D3**: 5 hidden layers (128, 96, 64, 48, 32 units)
- **D4**: 6 hidden layers (256, 192, 128, 96, 64 units)
- **D5**: 7 hidden layers (512, 384, 256, 192, 128 units) - largest
- **Input**: (32, 32, 3) CIFAR-10 images
- **Output**: 10 class logits
- **Activation**: ReLU, BatchNorm between layers
- **Batch Size**: 64

#### RNN (Recurrent Neural Networks) - R1 to R5
- **R1**: 1 RNN layer (hidden_size=128)
- **R2**: 2 RNN layers (hidden_size=256)
- **R3**: 3 RNN layers (hidden_size=384)
- **R4**: 4 RNN layers (hidden_size=512)
- **R5**: 5 RNN layers (hidden_size=640) - largest
- **Input**: (32, 32, 3) flattened to sequence
- **Output**: 10 class logits
- **Cell Type**: LSTM
- **Batch Size**: 32

#### CNN (Convolutional Neural Networks) - C1 to C5
- **C1**: 2 Conv blocks (16, 32 filters) + Dense
- **C2**: 3 Conv blocks (32, 64, 128 filters) + Dense
- **C3**: 4 Conv blocks (32, 64, 128, 256 filters) + Dense
- **C4**: 5 Conv blocks (32, 64, 128, 256, 512 filters) + Dense
- **C5**: 5 Conv blocks (64, 128, 256, 512, 512 filters) - largest
- **Input**: (32, 32, 3) CIFAR-10 images
- **Output**: 10 class logits
- **Convolution**: 3×3 kernels, ReLU activation
- **Pooling**: Max pooling 2×2
- **Batch Size**: 64

---

## Output Format

### File 1: Summary Statistics CSV
**Filename Pattern**: `benchmark_results_pytorch_cuda_YYYYMMDD_HHMMSS.csv`

**Columns (16 total)**:
| Column | Type | Description |
|--------|------|-------------|
| model | string | Model identifier (D1-D5, R1-R5, C1-C5) |
| model_family | string | Family: DNN, RNN, or CNN |
| trial_id | int | Trial number (1-N) |
| device | string | GPU name (e.g., "NVIDIA GeForce RTX 3060") |
| first_call_ms | float | Cold-start latency (milliseconds) |
| training_time_ms | float | Total training duration (milliseconds) |
| training_unit | string | Training configuration (epochs, dataset size, batch size) |
| inference_mean_ms | float | Mean inference latency across 100 samples |
| inference_std_ms | float | Standard deviation of latencies |
| inference_min_ms | float | Minimum latency observed |
| inference_max_ms | float | Maximum latency observed |
| inference_p50_ms | float | 50th percentile (median) |
| inference_p95_ms | float | 95th percentile |
| inference_p99_ms | float | 99th percentile |
| inference_iters | int | Number of measurements (100) |
| raw_times_count | int | Count of raw measurements captured (100) |
| backend | string | Backend: "PyTorch-CUDA" |

### File 2: Raw Timings CSV
**Filename Pattern**: `benchmark_results_pytorch_cuda_YYYYMMDD_HHMMSS_raw_timings.csv`

**Columns (4 total)**:
| Column | Type | Description |
|--------|------|-------------|
| model | string | Model identifier |
| trial_id | int | Trial number |
| measurement_idx | int | Sequential index (0-99) |
| time_ms | float | Individual inference latency (milliseconds) |

**Total Rows**: 1,500 (15 models × 100 measurements)

---

## Example Performance Results

### Inference Latency (ms) - Typical Values

| Model | Family | Mean | Std | Min | Max | P95 |
|-------|--------|------|-----|-----|-----|-----|
| D1 | DNN | 0.079 | 0.018 | 0.071 | 0.244 | 0.092 |
| D5 | DNN | 0.250 | 0.118 | 0.166 | 0.834 | 0.440 |
| R1 | RNN | 0.264 | 0.111 | 0.152 | 1.022 | 0.332 |
| R5 | RNN | 3.737 | 0.264 | 3.440 | 4.551 | 4.227 |
| C1 | CNN | 0.266 | 0.106 | 0.187 | 0.653 | 0.477 |
| C5 | CNN | 0.378 | 0.084 | 0.344 | 0.912 | 0.535 |

**Performance Tier**:
1. **Fastest**: D1 (DNN, 0.079 ms)
2. **Medium**: DNN/CNN (0.08-0.41 ms)
3. **Slowest**: R5 (RNN, 3.737 ms)

---

## Data Source

**Dataset**: CIFAR-10
- **Training Set**: 5,000 samples
- **Test Set**: 1,000 samples
- **Image Size**: 32×32 pixels
- **Channels**: RGB (3)
- **Classes**: 10
- **Loader**: PyTorch `torchvision.datasets.CIFAR10`

---

## Reproducibility Information

### GPU Optimization Flags
```python
torch.backends.cudnn.benchmark = True  # Enable autotuner
torch.cuda.synchronize()               # GPU-CPU synchronization
```

### Environment
- **CUDA Compute Capability**: 8.6 (RTX 3060)
- **Driver**: NVIDIA latest (WHQL certified)
- **PyTorch Build**: +cu121 (CUDA 12.1 specific)
- **Python Optimization**: No special flags

### Critical Notes for Reproducibility
1. **CUDA Version Must Match**: Install PyTorch for CUDA 12.1 specifically
2. **GPU Memory**: Requires 12 GB VRAM (RTX 3060 minimum)
3. **Individual Samples**: Inference measured on batch_size=1 (not batches)
4. **No Caching Artifacts**: Each measurement uses fresh GPU memory
5. **Warm GPU**: Warmup phase (Phase 4) ensures cache consistency

---

## Verification Command

Verify correct installation and GPU detection:

```bash
python -c "import torch; \
print(f'PyTorch: {torch.__version__}'); \
print(f'CUDA: {torch.version.cuda}'); \
print(f'GPU: {torch.cuda.get_device_name(0)}'); \
print(f'Compute Cap: {torch.cuda.get_device_capability(0)}')"
```

Expected output:
```
PyTorch: 2.5.1+cu121
CUDA: 12.1
GPU: NVIDIA GeForce RTX 3060
Compute Cap: (8, 6)
```

---

## Reference

- **PyTorch Official**: https://pytorch.org/
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
- **cuDNN**: https://developer.nvidia.com/cudnn
- **CIFAR-10**: https://www.cs.toronto.edu/~kriz/cifar.html