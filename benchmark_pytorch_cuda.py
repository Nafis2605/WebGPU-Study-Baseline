#!/usr/bin/env python3
"""
PyTorch CUDA Benchmark — Maximum Raw GPU Speed

All 15 baseline models (D1-D5, R1-R5, C1-C5) implemented natively in PyTorch
with direct CUDA integration.

Key advantages over TensorFlow/Numba version:
  - Direct CUDA via PyTorch (no Numba wrapper layer)
  - cuDNN autotuner: torch.backends.cudnn.benchmark = True
  - No graph-tracing / XLA overhead — PyTorch eager is faster at cold start
  - torch.cuda.synchronize() for microsecond-accurate GPU timing
  - No TF session overhead, no tf.keras eager dispatch cost

Usage:
    conda activate tf-gpu
    python benchmark_pytorch_cuda.py --trials 1
    python benchmark_pytorch_cuda.py --family RNN --trials 3
    python benchmark_pytorch_cuda.py --models D1 R2 C3 --trials 1
    python benchmark_pytorch_cuda.py --trials 5 --output results.csv
"""

import os
import sys
import time
import argparse
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# GPU ENFORCEMENT
# ============================================================================

def enforce_cuda():
    print("\n" + "=" * 80)
    print("[GPU CHECK] PyTorch CUDA Detection")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU found. PyTorch CUDA required.\n"
            "Install: pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print(f"[GPU CHECK] OK Device      : {props.name}")
    print(f"[GPU CHECK]   Compute Cap  : {props.major}.{props.minor}")
    print(f"[GPU CHECK]   VRAM         : {props.total_memory / 1024**3:.1f} GB")
    print(f"[GPU CHECK]   PyTorch      : {torch.__version__}")
    print(f"[GPU CHECK]   CUDA         : {torch.version.cuda}")

    # Max-performance cuDNN settings
    torch.backends.cudnn.benchmark = True   # auto-selects fastest cuDNN kernels
    torch.backends.cudnn.enabled   = True

    return device


# ============================================================================
# MODEL DEFINITIONS
# All models accept input shape (N, 32, 32, 3) — NHWC float32, on CUDA.
# ============================================================================

# ---- DNN ---------------------------------------------------------------

class D1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 64),  nn.ReLU(),
            nn.Linear(64,   10),
        )
    def forward(self, x): return self.net(x)


class D2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 128), nn.ReLU(),
            nn.Linear(128,  128), nn.ReLU(),
            nn.Linear(128,   10),
        )
    def forward(self, x): return self.net(x)


class D3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 256), nn.ReLU(),
            nn.Linear(256,  256), nn.ReLU(),
            nn.Linear(256,  256), nn.ReLU(),
            nn.Linear(256,   10),
        )
    def forward(self, x): return self.net(x)


class D4(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 512), nn.ReLU(),
            nn.Linear(512,  512), nn.ReLU(),
            nn.Linear(512,  512), nn.ReLU(),
            nn.Linear(512,  512), nn.ReLU(),
            nn.Linear(512,   10),
        )
    def forward(self, x): return self.net(x)


class D5(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024,   10),
        )
    def forward(self, x): return self.net(x)


# ---- CNN ---------------------------------------------------------------
# Shared conv block: (N,32,32,3) → permute NCHW →
#   Conv2d(3→64,3,pad=1)→ReLU→MaxPool2d(2) →  (N,64,16,16)
#   Conv2d(64→128,3,pad=1)→ReLU→MaxPool2d(2) → (N,128,8,8)
#   Flatten → 128*8*8 = 8192

class _CNNBase(nn.Module):
    def __init__(self, head: nn.Module):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,  64,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = head

    def forward(self, x):
        # NHWC → NCHW
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.head(self.features(x))


def C1():
    return _CNNBase(nn.Sequential(
        nn.Flatten(),
        nn.Linear(8192, 64),  nn.ReLU(),
        nn.Linear(64,   10),
    ))

def C2():
    return _CNNBase(nn.Sequential(
        nn.Flatten(),
        nn.Linear(8192, 128), nn.ReLU(),
        nn.Linear(128,  128), nn.ReLU(),
        nn.Linear(128,   10),
    ))

def C3():
    return _CNNBase(nn.Sequential(
        nn.Flatten(),
        nn.Linear(8192, 256), nn.ReLU(),
        nn.Linear(256,  256), nn.ReLU(),
        nn.Linear(256,  256), nn.ReLU(),
        nn.Linear(256,   10),
    ))

def C4():
    return _CNNBase(nn.Sequential(
        nn.Flatten(),
        nn.Linear(8192, 512), nn.ReLU(),
        nn.Linear(512,  512), nn.ReLU(),
        nn.Linear(512,  512), nn.ReLU(),
        nn.Linear(512,  512), nn.ReLU(),
        nn.Linear(512,   10),
    ))

def C5():
    return _CNNBase(nn.Sequential(
        nn.Flatten(),
        nn.Linear(8192, 1024), nn.ReLU(),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Linear(1024,   10),
    ))


# ---- RNN ---------------------------------------------------------------
# Reshape: (N,32,32,3) → (N,32,96)  [32 timesteps × 96 features]
# nonlinearity='relu' matches TF SimpleRNN(activation='relu')

class R1(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(96, 64, batch_first=True, nonlinearity='relu')
        self.fc  = nn.Linear(64, 10)

    def forward(self, x):
        out, _ = self.rnn(x.reshape(x.size(0), 32, 96))
        return self.fc(out[:, -1, :])


class R2(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(96,  128, batch_first=True, nonlinearity='relu')
        self.rnn2 = nn.RNN(128, 128, batch_first=True, nonlinearity='relu')
        self.fc   = nn.Linear(128, 10)

    def forward(self, x):
        x = x.reshape(x.size(0), 32, 96)
        out, _ = self.rnn1(x)
        out, _ = self.rnn2(out)
        return self.fc(out[:, -1, :])


class R3(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(96,  256, batch_first=True, nonlinearity='relu')
        self.rnn2 = nn.RNN(256, 256, batch_first=True, nonlinearity='relu')
        self.rnn3 = nn.RNN(256, 256, batch_first=True, nonlinearity='relu')
        self.fc   = nn.Linear(256, 10)

    def forward(self, x):
        x = x.reshape(x.size(0), 32, 96)
        out, _ = self.rnn1(x)
        out, _ = self.rnn2(out)
        out, _ = self.rnn3(out)
        return self.fc(out[:, -1, :])


class R4(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(96,  512, batch_first=True, nonlinearity='relu')
        self.rnn2 = nn.RNN(512, 512, batch_first=True, nonlinearity='relu')
        self.rnn3 = nn.RNN(512, 512, batch_first=True, nonlinearity='relu')
        self.rnn4 = nn.RNN(512, 512, batch_first=True, nonlinearity='relu')
        self.fc   = nn.Linear(512, 10)

    def forward(self, x):
        x = x.reshape(x.size(0), 32, 96)
        out, _ = self.rnn1(x)
        out, _ = self.rnn2(out)
        out, _ = self.rnn3(out)
        out, _ = self.rnn4(out)
        return self.fc(out[:, -1, :])


class R5(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(96,   1024, batch_first=True, nonlinearity='relu')
        self.rnn2 = nn.RNN(1024, 1024, batch_first=True, nonlinearity='relu')
        self.rnn3 = nn.RNN(1024, 1024, batch_first=True, nonlinearity='relu')
        self.rnn4 = nn.RNN(1024, 1024, batch_first=True, nonlinearity='relu')
        self.rnn5 = nn.RNN(1024, 1024, batch_first=True, nonlinearity='relu')
        self.fc   = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.reshape(x.size(0), 32, 96)
        out, _ = self.rnn1(x)
        out, _ = self.rnn2(out)
        out, _ = self.rnn3(out)
        out, _ = self.rnn4(out)
        out, _ = self.rnn5(out)
        return self.fc(out[:, -1, :])


# ============================================================================
# MODEL REGISTRY  {name: (factory, family, batch_size)}
# ============================================================================

MODEL_REGISTRY = {
    'D1': (D1, 'DNN', 64),
    'D2': (D2, 'DNN', 64),
    'D3': (D3, 'DNN', 64),
    'D4': (D4, 'DNN', 64),
    'D5': (D5, 'DNN', 64),
    'R1': (R1, 'RNN', 32),
    'R2': (R2, 'RNN', 32),
    'R3': (R3, 'RNN', 32),
    'R4': (R4, 'RNN', 32),
    'R5': (R5, 'RNN', 32),
    'C1': (C1, 'CNN', 64),
    'C2': (C2, 'CNN', 64),
    'C3': (C3, 'CNN', 64),
    'C4': (C4, 'CNN', 64),
    'C5': (C5, 'CNN', 64),
}

MODEL_ORDER = ['D1','D2','D3','D4','D5','R1','R2','R3','R4','R5','C1','C2','C3','C4','C5']


# ============================================================================
# DATA LOADING
# ============================================================================

def load_cifar10(train_size=5000, test_size=1000):
    """
    Load CIFAR-10 as numpy arrays (N,32,32,3) float32, labels int64.
    Tries torchvision first, falls back to keras (for data only).
    """
    print(f"[DATA] Loading CIFAR-10 (train={train_size}, test={test_size})...", end=" ", flush=True)

    try:
        import torchvision
        # torchvision .data attribute is already (N,32,32,3) uint8
        ds_train = torchvision.datasets.CIFAR10(
            root=os.path.expanduser("~/.cache/cifar10_pytorch"), train=True,  download=True
        )
        ds_test = torchvision.datasets.CIFAR10(
            root=os.path.expanduser("~/.cache/cifar10_pytorch"), train=False, download=True
        )
        x_tr = ds_train.data.astype(np.float32) / 255.0
        y_tr = np.array(ds_train.targets, dtype=np.int64)
        x_te = ds_test.data.astype(np.float32)  / 255.0
        y_te = np.array(ds_test.targets,  dtype=np.int64)
    except Exception:
        # Fallback: use keras purely as a CIFAR-10 download helper
        try:
            from tensorflow.keras.datasets import cifar10
        except ImportError:
            from keras.datasets import cifar10
        (x_tr, y_tr), (x_te, y_te) = cifar10.load_data()
        x_tr = x_tr.astype(np.float32) / 255.0
        x_te = x_te.astype(np.float32) / 255.0
        y_tr = y_tr.flatten().astype(np.int64)
        y_te = y_te.flatten().astype(np.int64)

    # Subsample
    idx_tr = np.random.choice(len(x_tr), train_size, replace=False)
    x_train, y_train = x_tr[idx_tr], y_tr[idx_tr]
    x_test,  y_test  = x_te[:test_size], y_te[:test_size]

    print(f"OK  x_train={x_train.shape}  x_test={x_test.shape}")
    return x_train, y_train, x_test, y_test


def to_gpu(x_np, y_np, device):
    """Push numpy arrays to GPU tensors."""
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    return x, y


# ============================================================================
# TIMING HELPER
# ============================================================================

def timed_forward(model, x):
    """
    Time a single forward pass with GPU synchronization.
    torch.cuda.synchronize() ensures GPU is done before reading the clock.
    Returns elapsed milliseconds.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


# ============================================================================
# BENCHMARK ENGINE
# ============================================================================

def benchmark_model(model_name, device, x_train, y_train, x_test,
                    trials=1, inference_iters=100, cooldown=2, capture_raw=True):

    factory, family, batch_size = MODEL_REGISTRY[model_name]
    results = []
    all_raw_times = {}  # Store raw times per trial

    # Pre-compute single-sample inference inputs (NOT batches) for raw timing
    test_size = x_test.size(0)
    inf_inputs = []
    for i in range(inference_iters):
        # Get individual sample, not batch of 32
        sample_idx = i % test_size
        inf_inputs.append(x_test[sample_idx:sample_idx + 1])  # Shape: (1, 32, 32, 3)

    for trial_id in range(1, trials + 1):
        print(f"\n[TRIAL {trial_id}/{trials}] Model: {model_name}")
        print("-" * 70)

        try:
            # ---- Phase 1: Model creation ----------------------------------------
            print(f"  Phase 1: Creating model {model_name}...", end=" ", flush=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model = factory().to(device)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            print(f"OK ({(t1-t0)*1000:.3f} ms)")

            # ---- Phase 2: Data already loaded -----------------------------------
            print(f"  Phase 2: Loading CIFAR-10 data...", end=" ", flush=True)
            print("OK")

            # ---- Phase 3: Warmup (8 untimed iterations - BEFORE measuring) -----
            print(f"  Phase 3: Warmup (8 untimed iterations)...", end=" ", flush=True)
            model.eval()
            with torch.no_grad():
                for i in range(8):
                    sample_idx = i % x_test.size(0)
                    _ = model(x_test[sample_idx:sample_idx + 1])
            torch.cuda.synchronize()
            print("OK")

            # ---- Phase 4: First inference (cold start - post-warmup) -----------
            print(f"  Phase 4: First inference (warmup-stabilized)...", end=" ", flush=True)
            first_input = x_test[:1]
            first_call_ms = timed_forward(model, first_input)
            print(f"OK ({first_call_ms:.3f} ms)")

            # ---- Phase 5: Training ----------------------------------------------
            print(f"  Phase 5: Training (EPOCHS=10, BATCH_SIZE={batch_size})...", end=" ", flush=True)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters())
            dataset   = TensorDataset(x_train, y_train)
            loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            model.train()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _epoch in range(10):
                for bx, by in loader:
                    optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()
                    out  = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            training_time_ms = (t1 - t0) * 1000.0
            print(f"OK ({training_time_ms:.1f} ms)")

            # ---- Phase 6: Inference (100 iterations, single samples) -----------
            print(f"  Phase 6: Inference ({inference_iters} iterations, PyTorch CUDA)...", end=" ", flush=True)
            model.eval()
            
            # Capture all raw inference times (per single sample)
            inference_times_raw = []
            for inp in inf_inputs:
                t = timed_forward(model, inp)
                inference_times_raw.append(t)
            
            # Compute statistics from raw values
            inference_times_np = np.array(inference_times_raw, dtype=np.float64)
            inf_mean = float(np.mean(inference_times_np))
            inf_std  = float(np.std(inference_times_np))
            inf_min  = float(np.min(inference_times_np))
            inf_max  = float(np.max(inference_times_np))
            inf_p50  = float(np.percentile(inference_times_np, 50))
            inf_p95  = float(np.percentile(inference_times_np, 95))
            inf_p99  = float(np.percentile(inference_times_np, 99))
            
            print(f"OK (mean: {inf_mean:.3f} ms, std: {inf_std:.3f} ms, min: {inf_min:.3f} ms, max: {inf_max:.3f} ms)")

            # ---- Phase 7: Cleanup -----------------------------------------------
            print(f"  Phase 7: Cleanup...", end=" ", flush=True)
            del model, optimizer, loader, dataset
            torch.cuda.empty_cache()
            print("OK")

            results.append({
                'model':            model_name,
                'model_family':     family,
                'trial_id':         trial_id,
                'device':           f'{torch.cuda.get_device_name(device)} (PyTorch CUDA)',
                'first_call_ms':    first_call_ms,
                'training_time_ms': training_time_ms,
                'training_unit':    f'model training EPOCHS=10 TRAIN_SIZE=5000 BATCH_SIZE={batch_size}',
                'inference_mean_ms':inf_mean,
                'inference_std_ms': inf_std,
                'inference_min_ms': inf_min,
                'inference_max_ms': inf_max,
                'inference_p50_ms': inf_p50,
                'inference_p95_ms': inf_p95,
                'inference_p99_ms': inf_p99,
                'inference_iters':  inference_iters,
                'backend':          'PyTorch-CUDA',
                'raw_times_count':  len(inference_times_raw),
            })
            
            # Store raw times if capture_raw is enabled
            if capture_raw:
                all_raw_times[f'{model_name}_trial{trial_id}'] = inference_times_raw
            print(f"\n  [OK] Trial {trial_id} complete")

        except Exception as e:
            import traceback
            print(f"\n  ERROR in trial {trial_id}: {e}")
            traceback.print_exc()
            results.append({
                'model': model_name, 'model_family': family, 'trial_id': trial_id,
                'device': 'ERROR', 'first_call_ms': -1, 'training_time_ms': -1,
                'training_unit': 'ERROR', 'inference_mean_ms': -1,
                'inference_std_ms': -1, 'inference_iters': -1, 'backend': 'ERROR',
            })

        if trial_id < trials:
            print(f"\n[COOLDOWN] Waiting {cooldown}s before next trial...")
            time.sleep(cooldown)

    return results, all_raw_times


# ============================================================================
# CSV OUTPUT
# ============================================================================

FIELDNAMES = [
    'model', 'model_family', 'trial_id', 'device',
    'first_call_ms', 'training_time_ms', 'training_unit',
    'inference_mean_ms', 'inference_std_ms', 
    'inference_min_ms', 'inference_max_ms',
    'inference_p50_ms', 'inference_p95_ms', 'inference_p99_ms',
    'inference_iters', 'raw_times_count', 'backend',
]


def write_results_csv(results, output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[CSV] Results written to: {output_file}")
    print(f"[CSV] Total rows: {len(results)}")


def write_raw_timing_data(all_raw_times, base_output_file):
    """
    Write raw individual inference timing measurements to a detailed CSV.
    One row per measurement, with model, trial, and time_ms columns.
    """
    if not all_raw_times:
        return
    
    raw_output_file = base_output_file.replace('.csv', '_raw_timings.csv')
    
    with open(raw_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'trial_id', 'measurement_idx', 'time_ms'])
        
        for key, times in all_raw_times.items():
            # Parse key format: 'MODEL_trialN'
            parts = key.rsplit('_trial', 1)
            model_name = parts[0]
            trial_id = int(parts[1]) if len(parts) > 1 else 1
            
            for idx, time_ms in enumerate(times):
                writer.writerow([model_name, trial_id, idx, f'{time_ms:.6f}'])
    
    print(f"[CSV] Raw timings written to: {raw_output_file}")
    print(f"[CSV] Total measurements: {sum(len(v) for v in all_raw_times.values())}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PyTorch CUDA Benchmark — Maximum Raw GPU Speed"
    )
    parser.add_argument('--trials',         type=int, default=1,
                        help='Trials per model (default: 1)')
    parser.add_argument('--family',         choices=['DNN', 'RNN', 'CNN'],
                        help='Benchmark one model family')
    parser.add_argument('--models',         nargs='+', metavar='MODEL',
                        help='Specific models (e.g. D1 R2 C3)')
    parser.add_argument('--output',         type=str, default=None,
                        help='Output CSV path (default: auto-timestamped)')
    parser.add_argument('--inference-iters',type=int, default=100,
                        help='Inference iterations per trial (default: 100)')
    parser.add_argument('--cooldown',       type=int, default=2,
                        help='Seconds between models (default: 2)')
    args = parser.parse_args()

    # Output filename
    if args.output is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'benchmark_results_pytorch_cuda_{ts}.csv'

    # GPU setup
    device = enforce_cuda()

    # Select models
    if args.models:
        models = args.models
    elif args.family:
        models = [m for m in MODEL_ORDER if MODEL_REGISTRY[m][1] == args.family]
    else:
        models = MODEL_ORDER

    total = len(models) * args.trials
    print(f"\n[BENCHMARK] {len(models)} model(s) × {args.trials} trial(s) = {total} runs")
    print(f"[BENCHMARK] Output: {args.output}")
    print(f"[BENCHMARK] Backend: PyTorch {torch.__version__} / CUDA {torch.version.cuda}")

    # Load data ONCE — reused across all models and trials
    x_train_np, y_train_np, x_test_np, _ = load_cifar10()
    x_train, y_train = to_gpu(x_train_np, y_train_np, device)
    x_test,  _       = to_gpu(x_test_np,  np.zeros(len(x_test_np), dtype=np.int64), device)

    # Run benchmark
    all_results = []
    all_raw_times_collected = {}
    for idx, model_name in enumerate(models, 1):
        _, family, _ = MODEL_REGISTRY[model_name]
        print(f"\n{'=' * 80}")
        print(f"[BENCHMARK] {idx}/{len(models)} - {model_name} ({family})")
        print("=" * 80)

        results, raw_times = benchmark_model(
            model_name, device,
            x_train, y_train, x_test,
            trials=args.trials,
            inference_iters=args.inference_iters,
            cooldown=args.cooldown,
            capture_raw=True,
        )
        all_results.extend(results)
        all_raw_times_collected.update(raw_times)

        if idx < len(models):
            print(f"\n[COOLDOWN] Waiting {args.cooldown}s before next model...")
            time.sleep(args.cooldown)

    # Write CSV
    write_results_csv(all_results, args.output)
    write_raw_timing_data(all_raw_times_collected, args.output)
    print(f"\n{'=' * 80}")
    print(f"[DONE] Benchmarking complete!")
    print(f"[DONE] Results saved to: {args.output}")
    print("=" * 80)


if __name__ == '__main__':
    main()
