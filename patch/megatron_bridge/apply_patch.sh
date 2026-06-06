#!/bin/bash

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
    echo "1. Try to auto-detect the appropriate patch based on installed Megatron-Bridge commit hash"
    echo "2. If no commit match, try to match based on Megatron-Bridge version"
    echo "3. If auto-detection fails, use the latest available patch file"
    echo ""
    echo "The script supports two types of patch files:"
    echo "  - Version-based: e.g., v0.1.0.patch, v0.2.0.patch"
    echo "  - Commit-based: e.g., 94d1870.patch, abcdef123.patch"
    echo ""
    echo "Examples:"
    echo "  $0                              # Auto-detect and apply appropriate patch"
    echo "  $0 --auto                       # Use latest patch file"
    echo "  $0 94d1870.patch               # Apply specific patch file"
    echo "  $0 --force 94d1870.patch       # Force apply specific patch"
}

# Function to list available patch files
list_patches() {
    echo "Available patch files in $SCRIPT_DIR:"
    echo ""

    # List version-based patches
    local version_patches=()
    local commit_patches=()

    for patch in "$SCRIPT_DIR"/*.patch; do
        if [ -f "$patch" ]; then
            local patch_basename=$(basename "$patch" .patch)
            if [[ "$patch_basename" =~ ^[a-f0-9]{7,40}$ ]]; then
                commit_patches+=("$(basename "$patch")")
            else
                version_patches+=("$(basename "$patch")")
            fi
        fi
    done

    if [ ${#version_patches[@]} -gt 0 ]; then
        echo "Version-based patches:"
        for patch in "${version_patches[@]}"; do
            echo "  - $patch"
        done
        echo ""
    fi

    if [ ${#commit_patches[@]} -gt 0 ]; then
        echo "Commit-based patches:"
        for patch in "${commit_patches[@]}"; do
            echo "  - $patch"
        done
        echo ""
    fi
}

# Function to get megatron-bridge version
get_megatron_bridge_version() {
    python -c "
import sys
import os
sys.stdout = open(os.devnull, 'w')
try:
    import megatron.bridge
    sys.stdout = sys.__stdout__
    if hasattr(megatron.bridge, '__version__'):
        print(megatron.bridge.__version__)
    else:
        print('')
except:
    sys.stdout = sys.__stdout__
    print('')
" 2>/dev/null || echo ""
}

# Function to get megatron-bridge commit hash
get_megatron_bridge_commit() {
    python -c "
import sys
import os
sys.stdout = open(os.devnull, 'w')
try:
    import megatron.bridge
    sys.stdout = sys.__stdout__
    # Try to get commit hash from __version__ if it contains git info
    if hasattr(megatron.bridge, '__version__'):
        version_str = megatron.bridge.__version__
        import re
        # Match patterns like 'v0.1.0+git.94d1870' or '+94d1870'
        commit_match = re.search(r'[+.]([a-f0-9]{7,40})', version_str)
        if commit_match:
            print(commit_match.group(1))
        else:
            # If no git info in version, try to get it from git
            import subprocess
            try:
                bridge_path = os.path.dirname(megatron.bridge.__file__)
                result = subprocess.run(['git', 'rev-parse', '--short=7', 'HEAD'],
                                      cwd=bridge_path, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print(result.stdout.strip())
                else:
                    print('')
            except:
                print('')
    else:
        # Try to get commit from git directly
        import subprocess
        try:
            bridge_path = os.path.dirname(megatron.bridge.__file__)
            result = subprocess.run(['git', 'rev-parse', '--short=7', 'HEAD'],
                                  cwd=bridge_path, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print('')
        except:
            print('')
except:
    sys.stdout = sys.__stdout__
    print('')
" 2>/dev/null || echo ""
}

# Function to find appropriate patch file
find_patch_file() {
    local bridge_version="$1"
    local bridge_commit="$2"
    local best_patch=""

    # First priority: Try exact commit hash match
    if [ -n "$bridge_commit" ]; then
        echo "Detected Megatron-Bridge commit: $bridge_commit" >&2

        # Try exact commit match (both full and short hash)
        for patch in "$SCRIPT_DIR"/*.patch; do
            if [ -f "$patch" ]; then
                local patch_name=$(basename "$patch" .patch)
                # Check if patch name is a commit hash (7-40 hex characters)
                if [[ "$patch_name" =~ ^[a-f0-9]{7,40}$ ]]; then
                    # Try exact match first
                    if [[ "$patch_name" == "$bridge_commit" ]]; then
                        echo "Found exact commit match: $(basename "$patch")" >&2
                        echo "$patch"
                        return 0
                    fi
                    # Try prefix match (e.g., 94d1870 matches 94d187012345)
                    if [[ "$bridge_commit" == "$patch_name"* ]] || [[ "$patch_name" == "$bridge_commit"* ]]; then
                        echo "Found commit prefix match: $(basename "$patch")" >&2
                        echo "$patch"
                        return 0
                    fi
                fi
            fi
        done
    fi

    # If version is available, try to find matching patch
    if [ -n "$bridge_version" ]; then
        echo "Detected Megatron-Bridge version: $bridge_version" >&2

        # Try exact version match first
        for patch in "$SCRIPT_DIR"/*.patch; do
            if [ -f "$patch" ]; then
                local patch_name=$(basename "$patch")
                # Skip commit hash patches for version matching
                local patch_basename=$(basename "$patch" .patch)
                if [[ ! "$patch_basename" =~ ^[a-f0-9]{7,40}$ ]]; then
                    if [[ "$patch_name" == *"$bridge_version"* ]]; then
                        echo "Found exact version match: $patch_name" >&2
                        echo "$patch"
                        return 0
                    fi
                fi
            fi
        done

        # Try partial version match (e.g., v0.1.x for v0.1.0)
        local major_minor=$(echo "$bridge_version" | sed -E 's/([0-9]+\.[0-9]+).*/\1/')
        for patch in "$SCRIPT_DIR"/*.patch; do
            if [ -f "$patch" ]; then
                local patch_name=$(basename "$patch")
                local patch_basename=$(basename "$patch" .patch)
                # Skip commit hash patches for version matching
                if [[ ! "$patch_basename" =~ ^[a-f0-9]{7,40}$ ]]; then
                    if [[ "$patch_name" == *"$major_minor"* ]]; then
                        echo "Found version match: $patch_name" >&2
                        echo "$patch"
                        return 0
                    fi
                fi
            fi
        done
    fi

    # Fall back to latest patch file (prefer version patches over commit patches)
    local latest_version_patch=""
    local latest_commit_patch=""

    for patch in $(ls -t "$SCRIPT_DIR"/*.patch 2>/dev/null); do
        if [ -f "$patch" ]; then
            local patch_basename=$(basename "$patch" .patch)
            if [[ "$patch_basename" =~ ^[a-f0-9]{7,40}$ ]]; then
                # This is a commit hash patch
                if [ -z "$latest_commit_patch" ]; then
                    latest_commit_patch="$patch"
                fi
            else
                # This is a version patch
                if [ -z "$latest_version_patch" ]; then
                    latest_version_patch="$patch"
                fi
            fi
        fi
    done

    # Prefer version patches over commit patches
    if [ -n "$latest_version_patch" ]; then
        echo "Using latest version patch file: $(basename "$latest_version_patch")" >&2
        echo "$latest_version_patch"
        return 0
    elif [ -n "$latest_commit_patch" ]; then
        echo "Using latest commit patch file: $(basename "$latest_commit_patch")" >&2
        echo "$latest_commit_patch"
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
        if [ -n "$PATCH_FILE_PATH" ]; then
            echo "Using latest patch file: $(basename "$PATCH_FILE_PATH")"
        fi
    else
        bridge_version=$(get_megatron_bridge_version)
        bridge_commit=$(get_megatron_bridge_commit)
        PATCH_FILE_PATH=$(find_patch_file "$bridge_version" "$bridge_commit")
    fi
fi

if [ ! -f "$PATCH_FILE_PATH" ]; then
    echo "Error: Patch file $PATCH_FILE_PATH does not exist."
    echo ""
    list_patches
    exit 1
fi

echo "Using patch file: $(basename "$PATCH_FILE_PATH")"

echo "Searching Megatron-Bridge install path..."

echo "Try to find Megatron-Bridge by python import..."
BRIDGE_PATH=$(python -c "
import sys
import os
sys.stdout = open(os.devnull, 'w')
try:
    import megatron.bridge
    sys.stdout = sys.__stdout__
    print(os.path.dirname(megatron.bridge.__file__))
except:
    sys.stdout = sys.__stdout__
    print('')
" 2>/dev/null || echo "")

if [ -n "$BRIDGE_PATH" ] && [ -d "$BRIDGE_PATH" ]; then
    # Navigate up from megatron/bridge/ to the repo root (src/megatron/bridge -> repo root)
    # The patch file paths start with src/megatron/bridge/...
    # So we need to find the directory that contains src/megatron/bridge/
    REPO_ROOT="$BRIDGE_PATH"
    # Walk up until we find a directory that contains src/ or setup.py/pyproject.toml
    while [ "$REPO_ROOT" != "/" ]; do
        if [ -f "$REPO_ROOT/pyproject.toml" ] || [ -f "$REPO_ROOT/setup.py" ] || [ -d "$REPO_ROOT/src/megatron/bridge" ]; then
            break
        fi
        REPO_ROOT=$(dirname "$REPO_ROOT")
    done

    if [ "$REPO_ROOT" = "/" ]; then
        # Fallback: assume the structure is .../src/megatron/bridge
        # So repo root is 3 levels up from BRIDGE_PATH
        REPO_ROOT=$(dirname "$(dirname "$(dirname "$BRIDGE_PATH")")")
    fi

    echo "Found Megatron-Bridge repo root: $REPO_ROOT"
else
    REPO_ROOT=""
fi

if [ -z "$REPO_ROOT" ] || [ ! -d "$REPO_ROOT" ]; then
    echo "Error: Could not find Megatron-Bridge installation path."
    echo "Please ensure that megatron-bridge is installed and try again."
    exit 1
fi

echo "Check if Megatron-Bridge is installed editably..."
IS_EDITABLE=false

if python -m pip list -e 2>/dev/null | grep -qi "megatron-bridge\|megatron_bridge"; then
    IS_EDITABLE=true
    echo "Megatron-Bridge is installed in editable mode."
fi

if [ "$IS_EDITABLE" = false ]; then
    echo "Warning: Megatron-Bridge is not installed in editable mode."
    echo "Please install Megatron-Bridge in editable mode using:"
    echo "  python -m pip install -e /path/to/Megatron-Bridge"
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
    if [ ! -f "$REPO_ROOT/$file" ]; then
        echo "Warning: Target file $REPO_ROOT/$file does not exist."
        echo "This may be normal if the file is new or the patch creates it."
    fi
done

echo "Applying patch to Megatron-Bridge..."
cd "$REPO_ROOT"

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
    echo "  2. The Megatron-Bridge version is incompatible with this patch"
    echo "  3. There are conflicting local changes"
    echo ""
    echo "You can try:"
    echo "  1. Check if the patch has already been applied"
    echo "  2. Use a different patch file for your Megatron-Bridge version"
    echo "  3. Resolve any conflicts manually"
    exit 1
fi

echo ""
echo "Patch applied successfully to the following files:"
for file in "${TARGET_FILES[@]}"; do
    if [ -f "$REPO_ROOT/$file" ]; then
        echo "  - $REPO_ROOT/$file"
    fi
done

echo ""
echo "Patch application completed successfully!"
echo "Patch file used: $(basename "$PATCH_FILE_PATH")"
