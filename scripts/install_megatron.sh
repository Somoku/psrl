#!/bin/bash

# Function: Install cuDNN and set CUDNN_PATH if not already installed or version is too low
install_cudnn() {
    local required_version="9.8.0.87"
    echo "Checking cuDNN installation..."

    # Check if nvidia-cudnn-cu12 is installed and get its version
    local installed_version=$(python -m pip show nvidia-cudnn-cu12 2>/dev/null | grep '^Version:' | awk '{print $2}')
    
    if [ -n "$installed_version" ]; then
        echo "Installed cuDNN version: $installed_version"
        # Compare versions
        if [ "$(printf '%s\n' "$required_version" "$installed_version" | sort -V | head -n1)" = "$required_version" ]; then
            echo "cuDNN version $installed_version is sufficient (>= $required_version)."
            set_cudnn_path
            return 0
        else
            echo "cuDNN version $installed_version is lower than required ($required_version). Proceeding to install..."
        fi
    else
        echo "cuDNN is not installed. Proceeding to install..."
    fi

    # Install cuDNN
    echo "Installing cuDNN Python package..."
    python -m pip install nvidia-cudnn-cu12=="$required_version"
    
    # Verify installation and set path
    set_cudnn_path
}

# Function: Set CUDNN_PATH after installation
set_cudnn_path() {
    local site_packages=$(python -c "import site; print(site.getsitepackages()[0])")
    local cudnn_path="$site_packages/nvidia/cudnn"
    
    if [ ! -d "$cudnn_path" ] || [ ! -f "$cudnn_path/include/cudnn_graph.h" ]; then
        echo "Error: cuDNN installation failed or path not found at $cudnn_path"
        exit 1
    fi
    
    echo "Setting CUDNN_PATH to: $cudnn_path"
    export CUDNN_PATH="$cudnn_path"
}

echo "1. Install cuDNN Python package"
install_cudnn

echo "2. Install TransformerEngine"
echo "Notice: TransformerEngine installation can take a long time, please be patient"
NVTE_FRAMEWORK=pytorch python -m pip install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2

echo "3. Install Megatron"
python -m pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.0rc3

# If you meet a weird bug from triton, please downgrade triton to 3.1.0
# See https://github.com/volcengine/verl/issues/904
# echo "4. Downgrade Triton"
# python -m pip install triton==3.1.0

echo "Successfully installed all packages for Megatron"