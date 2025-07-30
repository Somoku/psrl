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
    echo "1. Try to auto-detect the appropriate patch based on installed verl version"
    echo "2. If auto-detection fails, use the latest available patch file"
    echo ""
    echo "Examples:"
    echo "  $0                              # Auto-detect and apply appropriate patch"
    echo "  $0 --auto                       # Use latest patch file"
    echo "  $0 v0.5.x.patch                # Apply specific patch file"
    echo "  $0 --force v0.4.1.x.patch      # Force apply specific patch"
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

# Function to get verl version
get_verl_version() {
    python3 -c "
import sys
import os
sys.stdout = open(os.devnull, 'w')
try:
    import verl
    sys.stdout = sys.__stdout__
    print(verl.__version__)
except:
    sys.stdout = sys.__stdout__
    print('')
" 2>/dev/null || echo ""
}

# Function to find appropriate patch file
find_patch_file() {
    local verl_version="$1"
    local best_patch=""
    
    # If version is available, try to find matching patch
    if [ -n "$verl_version" ]; then
        echo "Detected verl version: $verl_version"
        
        # Try exact version match first
        for patch in "$SCRIPT_DIR"/*.patch; do
            if [ -f "$patch" ]; then
                local patch_name=$(basename "$patch")
                if [[ "$patch_name" == *"$verl_version"* ]]; then
                    echo "Found exact version match: $patch_name"
                    echo "$patch"
                    return 0
                fi
            fi
        done
        
        # Try partial version match (e.g., v0.5.x for v0.5.1)
        local major_minor=$(echo "$verl_version" | sed -E 's/([0-9]+\.[0-9]+).*/\1/')
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
        verl_version=$(get_verl_version)
        PATCH_FILE_PATH=$(find_patch_file "$verl_version")
    fi
fi

if [ ! -f "$PATCH_FILE_PATH" ]; then
    echo "Error: Patch file $PATCH_FILE_PATH does not exist."
    echo ""
    list_patches
    exit 1
fi

echo "Using patch file: $(basename "$PATCH_FILE_PATH")"

echo "Searching verl install path..."

echo "Try to find verl by python import..."
VERL_PATH=$(python3 -c "
import sys
import os
sys.stdout = open(os.devnull, 'w')
import verl
sys.stdout = sys.__stdout__
print(os.path.dirname(verl.__file__))
" 2>/dev/null || echo "")

if [ -n "$VERL_PATH" ] && [ -d "$VERL_PATH" ]; then
    VERL_PATH=$(dirname "$VERL_PATH")
    echo "Found verl path: $VERL_PATH"
else
    VERL_PATH=""
fi

if [ -z "$VERL_PATH" ] || [ ! -d "$VERL_PATH" ]; then
    echo "Error: Could not find verl installation path."
    echo "Please ensure that verl is installed and try again."
    exit 1
fi

echo "Found verl path: $VERL_PATH"

echo "Check if verl is installed editably..."
IS_EDITABLE=false

if pip list -e 2>/dev/null | grep -q "verl"; then
    IS_EDITABLE=true
    echo "verl is installed in editable mode."
fi

if [ "$IS_EDITABLE" = false ]; then
    echo "Error: verl is not installed in editable mode."
    echo "Please install verl in editable mode using:"
    echo "pip install -e /path/to/verl"
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
    if [ ! -f "$VERL_PATH/$file" ]; then
        echo "Warning: Target file $VERL_PATH/$file does not exist."
        echo "This may be normal if the file is new or the patch creates it."
    fi
done

echo "Applying patch to verl..."
cd "$VERL_PATH"

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
    echo "  2. The verl version is incompatible with this patch"
    echo "  3. There are conflicting local changes"
    echo ""
    echo "You can try:"
    echo "  1. Check if the patch has already been applied"
    echo "  2. Use a different patch file for your verl version"
    echo "  3. Resolve any conflicts manually"
    exit 1
fi

echo ""
echo "Patch applied successfully to the following files:"
for file in "${TARGET_FILES[@]}"; do
    if [ -f "$VERL_PATH/$file" ]; then
        echo "  - $VERL_PATH/$file"
    fi
done

echo ""
echo "Patch application completed successfully!"
echo "Patch file used: $(basename "$PATCH_FILE_PATH")"