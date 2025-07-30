#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [PATCH_FILE]"
    echo ""
    echo "Options:"
    echo "  --force        Apply patch even if not in editable mode"
    echo "  --auto         Automatically select the latest patch file"
    echo "  --list         List available patch files"
    echo "  -h, --help     Show this help message"
    echo ""
    echo "Arguments:"
    echo "  PATCH_FILE     Specific patch file to apply (optional)"
    echo ""
    echo "If no patch file is specified, the script will:"
    echo "1. Try to auto-detect the appropriate patch based on installed vllm version"
    echo "2. If auto-detection fails, use the latest available patch file"
    echo ""
    echo "Examples:"
    echo "  $0                              # Auto-detect and apply appropriate patch"
    echo "  $0 --auto                       # Use latest patch file"
    echo "  $0 v0.10.0.patch               # Apply specific patch file"
    echo "  $0 --force v0.9.0.1.patch      # Force apply specific patch"
}

# Function to list available patch files
list_patches() {
    echo "Available patch files in $SCRIPT_DIR:"
    for patch in "$SCRIPT_DIR"/*.patch; do
        if [ -f "$patch" ]; then
            echo "  - $(basename "$patch")"
        fi
    done
}

# Function to get vllm version
get_vllm_version() {
    python3 -c "
import sys
import os
sys.stdout = open(os.devnull, 'w')
try:
    import vllm
    sys.stdout = sys.__stdout__
    print(vllm.__version__)
except:
    sys.stdout = sys.__stdout__
    print('')
" 2>/dev/null || echo ""
}

# Function to find appropriate patch file
find_patch_file() {
    local vllm_version="$1"
    local best_patch=""
    
    # If version is available, try to find matching patch
    if [ -n "$vllm_version" ]; then
        echo "Detected vllm version: $vllm_version"
        
        # Try exact version match first
        for patch in "$SCRIPT_DIR"/*.patch; do
            if [ -f "$patch" ]; then
                local patch_name=$(basename "$patch")
                if [[ "$patch_name" == *"$vllm_version"* ]]; then
                    echo "Found exact version match: $patch_name"
                    echo "$patch"
                    return 0
                fi
            fi
        done
        
        # Try partial version match (e.g., v0.10.0 for v0.10.0.1)
        local major_minor_patch=$(echo "$vllm_version" | sed -E 's/([0-9]+\.[0-9]+\.[0-9]+).*/\1/')
        for patch in "$SCRIPT_DIR"/*.patch; do
            if [ -f "$patch" ]; then
                local patch_name=$(basename "$patch")
                if [[ "$patch_name" == *"$major_minor_patch"* ]]; then
                    echo "Found version match: $patch_name"
                    echo "$patch"
                    return 0
                fi
            fi
        done
        
        # Try major.minor match
        local major_minor=$(echo "$vllm_version" | sed -E 's/([0-9]+\.[0-9]+).*/\1/')
        for patch in "$SCRIPT_DIR"/*.patch; do
            if [ -f "$patch" ]; then
                local patch_name=$(basename "$patch")
                if [[ "$patch_name" == *"$major_minor"* ]]; then
                    echo "Found version match: $patch_name"
                    echo "$patch"
                    return 0
                fi
            fi
        done
    fi
    
    # Fall back to latest patch file
    local latest_patch=$(ls -t "$SCRIPT_DIR"/*.patch 2>/dev/null | head -n1)
    if [ -f "$latest_patch" ]; then
        echo "Using latest patch file: $(basename "$latest_patch")"
        echo "$latest_patch"
        return 0
    fi
    
    return 1
}

# Parse command line arguments
FORCE_MODE=false
AUTO_MODE=false
PATCH_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_MODE=true
            shift
            ;;
        --auto)
            AUTO_MODE=true
            shift
            ;;
        --list)
            list_patches
            exit 0
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *.patch)
            if [ -z "$PATCH_FILE" ]; then
                PATCH_FILE="$1"
            else
                echo "Error: Multiple patch files specified"
                exit 1
            fi
            shift
            ;;
        *)
            echo "Error: Unknown option $1"
            show_usage
            exit 1
            ;;
    esac
done

# Determine patch file to use
if [ -n "$PATCH_FILE" ]; then
    # Use specified patch file
    if [[ "$PATCH_FILE" == /* ]]; then
        # Absolute path
        PATCH_FILE_PATH="$PATCH_FILE"
    else
        # Relative path
        PATCH_FILE_PATH="$SCRIPT_DIR/$PATCH_FILE"
    fi
else
    # Auto-detect patch file
    if [ "$AUTO_MODE" = true ]; then
        PATCH_FILE_PATH=$(ls -t "$SCRIPT_DIR"/*.patch 2>/dev/null | head -n1)
    else
        vllm_version=$(get_vllm_version)
        PATCH_FILE_PATH=$(find_patch_file "$vllm_version")
    fi
fi

if [ ! -f "$PATCH_FILE_PATH" ]; then
    echo "Error: Patch file $PATCH_FILE_PATH does not exist."
    echo ""
    list_patches
    exit 1
fi

echo "Using patch file: $(basename "$PATCH_FILE_PATH")"

echo "Searching vllm install path..."

echo "Try to find vllm by python import..."

VLLM_PATH=$(python3 -c "
import sys
import os
sys.stdout = open(os.devnull, 'w')
import vllm
sys.stdout = sys.__stdout__
print(os.path.dirname(vllm.__file__))
" 2>/dev/null || echo "")

if [ -n "$VLLM_PATH" ] && [ -d "$VLLM_PATH" ]; then
    VLLM_PATH=$(dirname "$VLLM_PATH")
    echo "Found vllm path: $VLLM_PATH"
else
    VLLM_PATH=""
fi

if [ -z "$VLLM_PATH" ] || [ ! -d "$VLLM_PATH" ]; then
    echo "Error: Could not find vllm installation path."
    echo "Please ensure that vllm is installed and try again."
    exit 1
fi

echo "Found vllm path: $VLLM_PATH"

echo "Check if vllm is installed editably..."
IS_EDITABLE=false

if pip list -e 2>/dev/null | grep -q "vllm"; then
    IS_EDITABLE=true
    echo "vllm is installed in editable mode."
fi

if [ "$IS_EDITABLE" = false ]; then
    echo "Error: vllm is not installed in editable mode."
    echo "Please install vllm in editable mode using:"
    echo "pip install -e /path/to/vllm"
    echo ""
    echo "If you want to apply the patch anyway, please use --force option."
    if [ "$FORCE_MODE" != true ]; then
        exit 1
    else
        echo "Applying patch forcefully..."
    fi
fi

# Extract target files from the patch file
echo "Extracting target files from patch..."
TARGET_FILES=($(grep "^diff --git" "$PATCH_FILE_PATH" | sed 's/^diff --git a\/\(.*\) b\/.*$/\1/' | sort -u))

if [ ${#TARGET_FILES[@]} -eq 0 ]; then
    echo "Error: No target files found in patch file."
    exit 1
fi

echo "Target files found in patch:"
for file in "${TARGET_FILES[@]}"; do
    echo "  - $file"
done

# Verify target files exist
echo "Verifying target files exist..."
for file in "${TARGET_FILES[@]}"; do
    if [ ! -f "$VLLM_PATH/$file" ]; then
        echo "Warning: Target file $VLLM_PATH/$file does not exist."
        echo "This may be normal if the file is new or the patch creates it."
    fi
done

echo "Applying patch to vllm..."
cd "$VLLM_PATH"

if git apply --check "$PATCH_FILE_PATH" > /dev/null 2>&1; then
    echo "Patch is valid, applying..."
    if git apply "$PATCH_FILE_PATH"; then
        echo "Patch applied successfully."
    else
        echo "Error: Failed to apply patch."
        exit 1
    fi
else
    echo "Error: Patch is not valid or cannot be applied cleanly."
    echo "This might be because:"
    echo "  1. The patch has already been applied"
    echo "  2. The vllm version is incompatible with this patch"
    echo "  3. There are conflicting local changes"
    echo ""
    echo "You can try:"
    echo "  1. Check if the patch has already been applied"
    echo "  2. Use a different patch file for your vllm version"
    echo "  3. Resolve any conflicts manually"
    exit 1
fi

echo ""
echo "Patch applied successfully to the following files:"
for file in "${TARGET_FILES[@]}"; do
    if [ -f "$VLLM_PATH/$file" ]; then
        echo "  - $VLLM_PATH/$file"
    fi
done

echo ""
echo "Patch application completed successfully!"
echo "Patch file used: $(basename "$PATCH_FILE_PATH")"