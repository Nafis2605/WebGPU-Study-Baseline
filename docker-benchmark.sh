#!/bin/bash

# Docker GPU Benchmarking Quick Start Script
# Usage: ./docker-benchmark.sh [command] [args...]
# Commands: build, run, shell, logs, clean

set -e

IMAGE_NAME="benchmark-gpu"
IMAGE_TAG="latest"
CONTAINER_NAME="benchmark-gpu-native"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker first."
        echo "Download: https://www.docker.com/products/docker-desktop"
        exit 1
    fi
    log_info "Docker found: $(docker --version)"
}

# Check if GPU support is available
check_gpu_support() {
    log_info "Checking GPU support..."
    
    if ! docker run --rm --gpus all nvidia/cuda:11.2.2-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        log_warn "GPU support not available. You may need to:"
        echo "  1. Install NVIDIA GPU drivers"
        echo "  2. Install NVIDIA Container Toolkit"
        echo "  3. Restart Docker daemon"
        echo ""
        echo "See DOCKER_GPU_SETUP.md for detailed instructions."
        return 1
    fi
    
    log_info "GPU support verified ✓"
    return 0
}

# Build Docker image
build() {
    log_info "Building Docker image: $IMAGE_NAME:$IMAGE_TAG"
    log_info "This may take 5-10 minutes on first build..."
    
    docker build -t $IMAGE_NAME:$IMAGE_TAG .
    
    log_info "Build complete!"
    log_info "Image size: $(docker images --format '{{.Size}}' $IMAGE_NAME:$IMAGE_TAG)"
}

# Run benchmark
run() {
    local args="${@:-}"
    local trials="${args:-1}"
    
    log_info "Running benchmark with trials=$trials"
    log_info "Results will be saved to: ./results/"
    
    mkdir -p results
    
    if [ -z "$args" ]; then
        docker-compose run --rm benchmark-gpu \
            python3 benchmark_native.py --trials 1
    else
        docker-compose run --rm benchmark-gpu \
            python3 benchmark_native.py $args
    fi
    
    log_info "Benchmark complete!"
    log_info "Results: $(ls -lht results/benchmark_results_*.csv | head -1 | awk '{print $NF}')"
}

# Start interactive shell
shell() {
    log_info "Starting interactive shell in container..."
    log_info "Type 'exit' to return to host"
    
    mkdir -p results
    
    docker-compose run --rm benchmark-gpu bash
}

# Show container logs
logs() {
    log_info "Showing Docker logs..."
    docker-compose logs
}

# Clean up
clean() {
    log_warn "Cleaning up Docker resources..."
    
    docker-compose down 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    
    log_info "Cleanup complete"
    
    read -p "Remove Docker image? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker rmi $IMAGE_NAME:$IMAGE_TAG 2>/dev/null || true
        log_info "Image removed"
    fi
}

# Show help
show_help() {
    cat << EOF
Docker GPU Benchmarking Quick Start

Usage: $0 [command] [args...]

Commands:
  build         Build Docker image (required before first run)
  run [args]    Run benchmark in Docker
                  Examples:
                  ./docker-benchmark.sh run --trials 1
                  ./docker-benchmark.sh run --models D1 D2 --trials 3
  shell         Start interactive bash shell in container
  logs          Show container logs
  clean         Stop and remove containers (optionally remove image)
  check         Verify Docker and GPU setup
  help          Show this help message

Examples:
  # First time setup
  $0 build
  $0 check
  
  # Run all models with 1 trial
  $0 run --trials 1
  
  # Run specific models with 3 trials
  $0 run --models D1 D2 C1 --trials 3
  
  # Interactive shell
  $0 shell
  python3 benchmark_native.py --help

For detailed setup instructions, see DOCKER_GPU_SETUP.md
EOF
}

# Main logic
case "${1:-help}" in
    build)
        check_docker
        build
        ;;
    run)
        check_docker
        shift
        run "$@"
        ;;
    shell)
        check_docker
        shell
        ;;
    logs)
        check_docker
        logs
        ;;
    clean)
        clean
        ;;
    check)
        check_docker
        check_gpu_support && log_info "All checks passed! Ready to benchmark." || log_warn "GPU support check failed"
        ;;
    help)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
