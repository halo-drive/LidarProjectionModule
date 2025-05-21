import torch
import tensorrt as trt
import platform
import subprocess
import sys
import os
import shutil


def get_cuda_version():
    """Get CUDA version using nvcc"""
    nvcc_path = shutil.which('nvcc')
    if nvcc_path:
        try:
            result = subprocess.check_output([nvcc_path, '--version'], stderr=subprocess.STDOUT).decode()
            for line in result.split('\n'):
                if 'release' in line:
                    return line.split('release')[1].strip()
        except:
            pass
    return "nvcc not found in PATH"


def get_cudnn_info():
    """Get CUDNN information"""
    cudnn_paths = [
        '/usr/lib/x86_64-linux-gnu/libcudnn.so',
        '/usr/local/cuda/lib64/libcudnn.so'
    ]

    for path in cudnn_paths:
        if os.path.exists(path):
            return f"CUDNN found at {path}"

    # Check if it's in LD_LIBRARY_PATH
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    for directory in ld_library_path.split(':'):
        if directory and os.path.exists(os.path.join(directory, 'libcudnn.so')):
            return f"CUDNN found in LD_LIBRARY_PATH at {directory}"

    return "CUDNN not found in standard locations"


def main():
    print(f"System Platform: {platform.platform()}")
    print(f"Python version: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"TensorRT: {trt.__version__}")
    print(f"CUDA Toolkit: {get_cuda_version()}")
    print(f"CUDNN Status: {get_cudnn_info()}")

    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")

    # Print environment variables
    print("\nRelevant Environment Variables:")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")


if __name__ == "__main__":
    main()