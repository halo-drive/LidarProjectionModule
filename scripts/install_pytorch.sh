#!/bin/bash
# Install pre-built PyTorch if available

WHEELS_DIR="/workspace/pytorch_builds/wheels"

if [ -f "$WHEELS_DIR"/torch-2.1.0-*.whl ]; then
    echo "Installing pre-built PyTorch wheel..."
    pip3 install "$WHEELS_DIR"/torch-2.1.0-*.whl --force-reinstall

    # Also install torchvision if available
    if [ -f "$WHEELS_DIR"/torchvision-*.whl ]; then
        pip3 install "$WHEELS_DIR"/torchvision-*.whl --force-reinstall
    fi

    echo "PyTorch installation complete!"
    python3 -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"
else
    echo "No pre-built PyTorch wheel found. Run scripts/build_pytorch.sh to build from source."
fi