# Docker GPU Benchmarking Quick Start Script (PowerShell)
# Usage: .\docker-benchmark.ps1 [command] [args...]
# Commands: build, run, shell, logs, clean

param(
    [string]$Command = "help",
    [string[]]$Args = @()
)

# Configuration
$IMAGE_NAME = "benchmark-gpu"
$IMAGE_TAG = "latest"
$CONTAINER_NAME = "benchmark-gpu-native"

# Color codes
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO]  $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN]  $Message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if Docker is installed
function Test-Docker {
    try {
        $null = docker --version
        Write-Info "Docker found: $(docker --version)"
        return $true
    }
    catch {
        Write-Error-Custom "Docker not found. Please install Docker Desktop for Windows."
        Write-Host "Download: https://www.docker.com/products/docker-desktop"
        return $false
    }
}

# Check if GPU support is available
function Test-GPUSupport {
    Write-Info "Checking GPU support..."
    
    try {
        $null = docker run --rm --gpus all nvidia/cuda:11.2.2-base-ubuntu20.04 nvidia-smi 2>$null
        Write-Info "GPU support verified ✓"
        return $true
    }
    catch {
        Write-Warn "GPU support not available. You may need to:"
        Write-Host "  1. Install NVIDIA GPU drivers"
        Write-Host "  2. Install NVIDIA Container Toolkit"
        Write-Host "  3. Restart Docker daemon"
        Write-Host ""
        Write-Host "See DOCKER_GPU_SETUP.md for detailed instructions."
        return $false
    }
}

# Build Docker image
function Invoke-Build {
    Write-Info "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
    Write-Info "This may take 5-10 minutes on first build..."
    
    docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .
    
    if ($LASTEXITCODE -eq 0) {
        Write-Info "Build complete!"
        $size = docker images --format "{{.Size}}" "${IMAGE_NAME}:${IMAGE_TAG}"
        Write-Info "Image size: $size"
    }
    else {
        Write-Error-Custom "Build failed!"
        exit 1
    }
}

# Run benchmark
function Invoke-Run {
    param([string[]]$BenchmarkArgs)
    
    if ($BenchmarkArgs.Count -eq 0) {
        $BenchmarkArgs = @("--trials", "1")
    }
    
    Write-Info "Running benchmark with args: $($BenchmarkArgs -join ' ')"
    Write-Info "Results will be saved to: ./results/"
    
    if (!(Test-Path "results")) {
        New-Item -ItemType Directory -Path "results" -ErrorAction SilentlyContinue | Out-Null
    }
    
    docker-compose run --rm benchmark-gpu python3 benchmark_native.py @BenchmarkArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Info "Benchmark complete!"
        $latestResult = Get-ChildItem "results/benchmark_results_*.csv" -ErrorAction SilentlyContinue | 
                        Sort-Object LastWriteTime -Descending | 
                        Select-Object -First 1
        if ($latestResult) {
            Write-Info "Results: $($latestResult.FullName)"
        }
    }
    else {
        Write-Error-Custom "Benchmark failed!"
        exit 1
    }
}

# Start interactive shell
function Invoke-Shell {
    Write-Info "Starting interactive shell in container..."
    Write-Info "Type 'exit' to return to host"
    
    if (!(Test-Path "results")) {
        New-Item -ItemType Directory -Path "results" -ErrorAction SilentlyContinue | Out-Null
    }
    
    docker-compose run --rm benchmark-gpu bash
}

# Show logs
function Invoke-Logs {
    Write-Info "Showing Docker logs..."
    docker-compose logs
}

# Clean up
function Invoke-Clean {
    Write-Warn "Cleaning up Docker resources..."
    
    docker-compose down 2>$null
    docker rm $CONTAINER_NAME 2>$null
    
    Write-Info "Stopped and removed containers"
    
    $response = Read-Host "Remove Docker image? (y/N)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        docker rmi "${IMAGE_NAME}:${IMAGE_TAG}" 2>$null
        Write-Info "Image removed"
    }
    
    Write-Info "Cleanup complete"
}

# Show help
function Show-Help {
    $helpText = @"
Docker GPU Benchmarking Quick Start

Usage: .\docker-benchmark.ps1 [command] [args...]

Commands:
  build         Build Docker image (required before first run)
  run [args]    Run benchmark in Docker
                  Examples:
                  .\docker-benchmark.ps1 run --trials 1
                  .\docker-benchmark.ps1 run --models D1 D2 --trials 3
  shell         Start interactive bash shell in container
  logs          Show container logs
  clean         Stop and remove containers (optionally remove image)
  check         Verify Docker and GPU setup
  help          Show this help message

Examples:
  # First time setup
  .\docker-benchmark.ps1 build
  .\docker-benchmark.ps1 check
  
  # Run all models with 1 trial
  .\docker-benchmark.ps1 run --trials 1
  
  # Run specific models with 3 trials
  .\docker-benchmark.ps1 run --models D1 D2 C1 --trials 3
  
  # Interactive shell
  .\docker-benchmark.ps1 shell
  python3 benchmark_native.py --help

For detailed setup instructions, see DOCKER_GPU_SETUP.md
"@
    Write-Host $helpText
}

# Main logic
switch ($Command.ToLower()) {
    "build" {
        if (Test-Docker) {
            Invoke-Build
        }
    }
    "run" {
        if (Test-Docker) {
            Invoke-Run $Args
        }
    }
    "shell" {
        if (Test-Docker) {
            Invoke-Shell
        }
    }
    "logs" {
        if (Test-Docker) {
            Invoke-Logs
        }
    }
    "clean" {
        if (Test-Docker) {
            Invoke-Clean
        }
    }
    "check" {
        if (Test-Docker) {
            if (Test-GPUSupport) {
                Write-Info "All checks passed! Ready to benchmark."
            }
            else {
                Write-Warn "GPU support check failed"
            }
        }
    }
    "help" {
        Show-Help
    }
    default {
        Write-Error-Custom "Unknown command: $Command"
        Write-Host ""
        Show-Help
        exit 1
    }
}
