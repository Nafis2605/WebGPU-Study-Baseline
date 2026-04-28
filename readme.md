# Baseline Model Experiments

15 baseline neural network models for GPU benchmarking (5 DNN, 5 RNN, 5 CNN).

**Models**: D1–D5 (DNN), R1–R5 (RNN), C1–C5 (CNN)

---

## Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (RTX 3060+ recommended)
- conda (optional but recommended)

### Setup

```bash
# Create conda environment
conda create -n tf-gpu python=3.10 -y
conda activate tf-gpu

# Install dependencies
pip install -r requirements.txt
```

**Key packages**: TensorFlow 2.13+, Numba 0.65+, NumPy, pandas

---

## Benchmarking

### Quick Start

```bash
# Run all 15 models (1 trial)
python benchmark_numba_cuda.py --trials 1

# Run specific family
python benchmark_numba_cuda.py --family DNN --trials 1

# Run specific models
python benchmark_numba_cuda.py --models D1 D2 R1 --trials 1

# Multiple trials
python benchmark_numba_cuda.py --trials 5

# Custom output
python benchmark_numba_cuda.py --trials 1 --output results.csv
```

### Command Arguments
- `--trials N`: Number of trials per model
- `--family {DNN,RNN,CNN}`: Benchmark specific family
- `--models MODEL [MODEL ...]`: Specific models by name
- `--output PATH`: Output CSV path

### Output

CSV format with columns:
```
model, model_family, trial_id, device, first_call_ms, training_time_ms, 
training_unit, inference_mean_ms, inference_std_ms, inference_iters, backend
```

**Metrics**:
- **first_call_ms**: Cold-start latency
- **training_time_ms**: Total training time (10 epochs)
- **inference_mean_ms**: Mean latency across 100 inferences
- **inference_std_ms**: Standard deviation

---

## Web Interface

Run trained models in browser:

```bash
node server.js
```

Access at `http://localhost:5500` and select model in `index.html`

---

## Repository Structure

```
├── CNN/               (C1.js, C2.js, C3.js, C4.js, C5.js)
├── DNN/               (D1.js, D2.js, D3.js, D4.js, D5.js)
├── RNN/               (R1.js, R2.js, R3.js, R4.js, R5.js)
├── data/              (CIFAR-10 data loaders)
├── benchmark_numba_cuda.py  (GPU benchmark engine)
├── requirements.txt
├── index.html
├── server.js
└── README.md
```