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
    echo "1. Try to auto-detect the appropriate patch based on installed verl commit hash"
    echo "2. If no commit match, try to match based on verl version"
    echo "3. If auto-detection fails, use the latest available version patch file"
    echo ""
    echo "The script supports two types of patch files:"
    echo "  - Version-based: e.g., v0.5.x.patch, v0.4.1.patch"
    echo "  - Commit-based: e.g., 5c98ed1.patch, abcdef123.patch"
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

# Function to get verl version
get_verl_version() {
    python -c "
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

# Function to get verl commit hash
get_verl_commit() {
    python -c "
import sys
import os
sys.stdout = open(os.devnull, 'w')
try:
    import verl
    sys.stdout = sys.__stdout__
    # Try to get commit hash from __version__ if it contains git info
    if hasattr(verl, '__version__'):
        version_str = verl.__version__
        # Look for git commit hash pattern in version string
        import re
        # Match patterns like 'v0.5.0+git.abcdef1' or 'abcdef1'
        commit_match = re.search(r'[+.]([a-f0-9]{7,40})', version_str)
        if commit_match:
            print(commit_match.group(1))
        else:
            # If no git info in version, try to get it from git
            import subprocess
            try:
                verl_path = os.path.dirname(verl.__file__)
                result = subprocess.run(['git', 'rev-parse', '--short=7', 'HEAD'], 
                                      cwd=verl_path, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print(result.stdout.strip())
                else:
                    print('')
            except:
                print('')
    else:
        print('')
except:
    sys.stdout = sys.__stdout__
    print('')
" 2>/dev/null || echo ""
}

# Function to find appropriate patch file
find_patch_file() {
    local verl_version="$1"
    local verl_commit="$2"
    local best_patch=""
    
    # First priority: Try exact commit hash match
    if [ -n "$verl_commit" ]; then
        echo "Detected verl commit: $verl_commit" >&2
        
        # Try exact commit match (both full and short hash)
        for patch in "$SCRIPT_DIR"/*.patch; do
            if [ -f "$patch" ]; then
                local patch_name=$(basename "$patch" .patch)
                # Check if patch name is a commit hash (7-40 hex characters)
                if [[ "$patch_name" =~ ^[a-f0-9]{7,40}$ ]]; then
                    # Try exact match first
                    if [[ "$patch_name" == "$verl_commit" ]]; then
                        echo "Found exact commit match: $(basename "$patch")" >&2
                        echo "$patch"
                        return 0
                    fi
                    # Try prefix match (e.g., 5c98ed1 matches 5c98ed1234567)
                    if [[ "$verl_commit" == "$patch_name"* ]] || [[ "$patch_name" == "$verl_commit"* ]]; then
                        echo "Found commit prefix match: $(basename "$patch")" >&2
                        echo "$patch"
                        return 0
                    fi
                fi
            fi
        done
    fi

    # If version is available, try to find matching patch
    if [ -n "$verl_version" ]; then
        echo "Detected verl version: $verl_version" >&2
        
        # Try exact version match first
        for patch in "$SCRIPT_DIR"/*.patch; do
            if [ -f "$patch" ]; then
                local patch_name=$(basename "$patch")
                # Skip commit hash patches for version matching
                local patch_basename=$(basename "$patch" .patch)
                if [[ ! "$patch_basename" =~ ^[a-f0-9]{7,40}$ ]]; then
                    if [[ "$patch_name" == *"$verl_version"* ]]; then
                        echo "Found exact version match: $patch_name" >&2
                        echo "$patch"
                        return 0
                    fi
                fi
            fi
        done
        
        # Try partial version match (e.g., v0.5.x for v0.5.1)
        local major_minor=$(echo "$verl_version" | sed -E 's/([0-9]+\.[0-9]+).*/\1/')
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
        verl_version=$(get_verl_version)
        verl_commit=$(get_verl_commit)
        PATCH_FILE_PATH=$(find_patch_file "$verl_version" "$verl_commit")
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
VERL_PATH=$(python -c "
import sys
import os
sys.stdout = open(os.devnull, 'w')
try:
    import verl
    sys.stdout = sys.__stdout__
    print(os.path.dirname(verl.__file__))
except:
    sys.stdout = sys.__stdout__
    print('')
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