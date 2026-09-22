#!/usr/bin/env bash
# Bake a per-problem grader image with the dependencies its repository needs.
# Usage: `repair_grader_images.sh --parquet PATH [--instances REGEX]`
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPARE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$(dirname "$PREPARE_DIR")")")"

PARQUET=""
INSTANCES=""
IMAGE_DIR="${PSRL_GRADER_IMAGE_DIR:-${PSRL_WORKSPACE:-.}/docker_images/swe_gym_293}"
NAMESPACE="${PSRL_GRADER_IMAGE_NAMESPACE:-psrl/swebench-grader}"
REVISION="${PSRL_GRADER_IMAGE_REVISION:-v1}"
PROXY="${PSRL_GRADER_PROXY:-}"
APT_MIRROR="${PSRL_GRADER_APT_MIRROR:-}"
# The interpreter must be able to import pandas, which a bare ssh PATH may miss.
PYTHON="${PSRL_GRADER_PYTHON:-python3}"
MAP=""
SAVE_TARS=0
APPLY=1
DRY_RUN=0
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --parquet)    PARQUET="$2"; shift 2 ;;
        --instances)  INSTANCES="$2"; shift 2 ;;
        --image-dir)  IMAGE_DIR="$2"; shift 2 ;;
        --namespace)  NAMESPACE="$2"; shift 2 ;;
        --revision)   REVISION="$2"; shift 2 ;;
        --proxy)      PROXY="$2"; shift 2 ;;
        --apt-mirror) APT_MIRROR="$2"; shift 2 ;;
        --map)        MAP="$2"; shift 2 ;;
        --save-tars)  SAVE_TARS=1; shift ;;
        --no-apply)   APPLY=0; shift ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --force)      FORCE=1; shift ;;
        -h|--help)    sed -n '2,3p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$PARQUET" && -f "$PARQUET" ]] || { echo "ERROR: --parquet FILE is required." >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found." >&2; exit 1; }

[[ -n "$MAP" ]] || MAP="${PARQUET%.parquet}.grader_images.tsv"

# A repaired tag is content addressed on the base image, the recipe, and the
# revision. Editing one recipe then only invalidates the images it changes.
digest() { printf '%s|%s|%s' "$1" "$2" "$3" | sha256sum | cut -c1-12; }

# Single definition of the derivative tag, so the bake and the tar export agree.
repaired_tag() { printf '%s:%s' "$NAMESPACE" "$(digest "$1" "$2" "$REVISION")"; }

# Tar names must match `load_all_nodes.sh`, which maps `/` and `:` to `__`.
tar_name() { printf '%s' "$1" | sed 's#/#__#g; s#:#__#g'; }

# Recipes are keyed on the problem's repository. An unlisted repository is left
# alone, because its failure needs a data fix rather than an image fix.
recipe_for_repo() {
    case "$1" in
        pvlib/pvlib-python) printf 'numpy1\n' ;;
        pyvista/pyvista)    printf 'gl\n' ;;
        *)                  printf '\n' ;;
    esac
}

# Each recipe ends by importing the repository, so a bake that repaired nothing
# is never committed.
recipe_script() {
    case "$1" in
        numpy1) cat <<'RECIPE'
set -e
PY=/opt/miniconda3/envs/testbed/bin/python
"$PY" -m pip install --no-cache-dir --disable-pip-version-check ${PSRL_GRADER_PIP_ARGS:-} "numpy<2"
"$PY" -c "import pvlib"
RECIPE
            ;;
        gl) cat <<'RECIPE'
set -e
export DEBIAN_FRONTEND=noninteractive
if [ -n "${PSRL_GRADER_APT_MIRROR:-}" ]; then
  sed -i -E "s#https?://(archive|security)\.ubuntu\.com/ubuntu#${PSRL_GRADER_APT_MIRROR}#g" /etc/apt/sources.list
fi
apt-get update -qq
apt-get install -y -qq --no-install-recommends libgl1 libglx-mesa0 libxrender1 libxext6 libsm6 libice6 \
  || apt-get install -y -qq --no-install-recommends libgl1-mesa-glx libxrender1 libxext6 libsm6 libice6
/opt/miniconda3/envs/testbed/bin/python -c "import pyvista"
RECIPE
            ;;
        *) return 1 ;;
    esac
}

plan() {
    local args=(--parquet "$PARQUET" --plan)
    [ -n "$INSTANCES" ] && args+=(--instances "$INSTANCES")
    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON" -m examples.mini_swe.prepare.retarget_problem_images "${args[@]}"
}

# Report progress on stderr so stdout carries only the `base<TAB>repaired` line.
repair_one() {
    local base_image="$1" recipe="$2"
    local tag
    tag="$(repaired_tag "$base_image" "$recipe")"

    if docker image inspect "$tag" >/dev/null 2>&1 && [ "$FORCE" != "1" ]; then
        echo "skip (already repaired): $tag" >&2
        printf '%s\t%s\n' "$base_image" "$tag"
        return 0
    fi
    if [ "$DRY_RUN" = "1" ]; then
        echo "would repair $base_image with recipe=$recipe -> $tag" >&2
        printf '%s\t%s\n' "$base_image" "$tag"
        return 0
    fi

    docker image inspect "$base_image" >/dev/null 2>&1 || docker pull "$base_image"
    local container_name="psrl-grader-repair-$$"
    local cid exec_env=()
    [ -n "${PSRL_GRADER_PIP_ARGS:-}" ] && exec_env=(-e "PSRL_GRADER_PIP_ARGS=${PSRL_GRADER_PIP_ARGS}")
    [ -n "$APT_MIRROR" ] && exec_env+=(-e "PSRL_GRADER_APT_MIRROR=$APT_MIRROR")
    # Task sandboxes have no egress, so a proxy is opt in and never a default.
    if [ -n "$PROXY" ]; then
        exec_env+=(-e "http_proxy=$PROXY" -e "https_proxy=$PROXY" -e "HTTP_PROXY=$PROXY" -e "HTTPS_PROXY=$PROXY")
        [ -n "${no_proxy:-}" ] && exec_env+=(-e "no_proxy=$no_proxy" -e "NO_PROXY=$no_proxy")
    fi

    cid="$(docker create --name "$container_name" "$base_image" tail -f /dev/null)"
    docker start "$cid" >/dev/null
    # The recipe output goes to stderr, because stdout carries only the mapping.
    if ! docker exec "${exec_env[@]}" "$cid" bash -lc "$(recipe_script "$recipe")" >&2; then
        docker rm -f "$container_name" >/dev/null 2>&1 || true
        echo "ERROR: recipe $recipe failed for $base_image." >&2
        return 1
    fi
    docker commit "$container_name" "$tag" >/dev/null
    docker rm -f "$container_name" >/dev/null 2>&1 || true
    echo "repaired: $tag" >&2
    printf '%s\t%s\n' "$base_image" "$tag"
}

if [ "$DRY_RUN" = "1" ]; then
    echo "=== repair plan (dry run) ==="
else
    echo "=== repairing grader images ==="
fi

mapfile -t PLAN < <(plan)
[[ ${#PLAN[@]} -gt 0 ]] || { echo "ERROR: no rows matched --parquet/--instances." >&2; exit 1; }

IMAGES_LIST="${MAP%.tsv}.images.txt"
repaired=0
failed=0
if [ "$DRY_RUN" != "1" ]; then
    : > "$MAP"
    : > "$IMAGES_LIST"
fi
for line in "${PLAN[@]}"; do
    IFS=$'\t' read -r base_image repo instances <<<"$line"
    recipe="$(recipe_for_repo "$repo")"
    if [ -z "$recipe" ]; then
        echo "skip (no recipe): $repo ($base_image)"
        continue
    fi
    if mapping="$(repair_one "$base_image" "$recipe")"; then
        if [ "$(printf '%s\n' "$mapping" | wc -l)" != "1" ] || [[ "$mapping" != *$'\t'* ]]; then
            echo "ERROR: unexpected repair output for $base_image." >&2
            failed=$((failed + 1))
            continue
        fi
        if [ "$DRY_RUN" != "1" ]; then
            printf '%s\n' "$mapping" >> "$MAP"
            printf '%s\n' "${mapping#*$'\t'}" >> "$IMAGES_LIST"
        fi
        repaired=$((repaired + 1))
    else
        failed=$((failed + 1))
    fi
    if [ "$SAVE_TARS" = "1" ] && [ "$DRY_RUN" != "1" ]; then
        tag="$(repaired_tag "$base_image" "$recipe")"
        mkdir -p "$IMAGE_DIR"
        tar_path="$IMAGE_DIR/$(tar_name "$tag").tar"
        if [ ! -f "$tar_path" ]; then
            echo "saving $tar_path" >&2
            docker save "$tag" -o "$tar_path"
        fi
    fi
done

echo
if [ "$DRY_RUN" = "1" ]; then
    echo "would repair $repaired image(s), $failed planned failure(s)"
    exit 0
fi
echo "repaired $repaired image(s), failed $failed, map written to $MAP"
if [ "$APPLY" = "1" ]; then
    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON" -m examples.mini_swe.prepare.retarget_problem_images \
        --parquet "$PARQUET" --overrides "$MAP"
fi
if [ "$SAVE_TARS" = "1" ]; then
    echo "Run: bash $SCRIPT_DIR/load_all_nodes.sh --hosts <hosts-file> \\"
    echo "       --image-dir $IMAGE_DIR --images-list $IMAGES_LIST"
fi
[ "$failed" = "0" ]
