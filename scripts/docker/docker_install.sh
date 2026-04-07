#!/bin/bash
# set -v

# Save a Docker image as a tar (for docker load / docker_copy.sh).
# Args or env: $1/DOCKER_IMAGE_DIR, $2/DOCKER_IMAGE_FILE (tar basename), $3/DOCKER_IMAGE_REF (e.g. python:3.11-slim).
# Legacy env alias: DOCKER_IMAGE_NAME (same as DOCKER_IMAGE_FILE if DOCKER_IMAGE_FILE is unset).
#
# DOCKER_INSTALL_METHOD:
#   docker (default) — docker pull + docker save (needs Docker daemon).
#   skopeo            — registry -> tar in one step, no docker pull (needs skopeo).
#   crane             — same idea (needs crane).
#
# Example:
#   ./docker_install.sh /data/docker_images python_3.11-slim.tar python:3.11-slim
#   DOCKER_INSTALL_METHOD=skopeo ./docker_install.sh /data img.tar python:3.11-slim
# Then copy/load on nodes: ./docker_copy.sh "$DOCKER_NODE_IPS" "$DOCKER_NODE_NUM" /data/docker_images python_3.11-slim.tar
#
# Pulling via a Docker Hub mirror (when registry-1.docker.io is slow or blocked):
#   DOCKERHUB_MIRROR=docker.m.daocloud.io ./docker_install.sh /data img.tar python:3.11-slim
# Or pass a full ref (no DOCKERHUB_MIRROR needed):
#   ./docker_install.sh /data img.tar docker.m.daocloud.io/library/python:3.11-slim
# Registry mirrors for the daemon: /etc/docker/daemon.json "registry-mirrors", then restart docker.

# If DOCKERHUB_MIRROR is set, rewrite short Docker Hub refs to pull from that host (skopeo/docker/crane).
apply_dockerhub_mirror() {
    local ref="$1"
    local m="${DOCKERHUB_MIRROR:-}"
    [ -n "$m" ] || { printf '%s' "$ref"; return 0; }
    m="${m#/}"; m="${m%/}"
    case "$ref" in
    "$m"/*|docker.io/*) printf '%s' "$ref"; return 0 ;;
    esac
    local first="${ref%%/*}"
    if [[ "$ref" == */* ]]; then
        if [[ "$first" == *.* || "$first" == localhost* ]]; then
            printf '%s' "$ref"
        else
            printf '%s/%s' "$m" "$ref"
        fi
    else
        printf '%s/library/%s' "$m" "$ref"
    fi
}

[ -n "$1" ] && DOCKER_IMAGE_DIR="$1"
[ -n "$2" ] && DOCKER_IMAGE_FILE="$2"
[ -n "$3" ] && DOCKER_IMAGE_REF="$3"

DOCKER_IMAGE_FILE=${DOCKER_IMAGE_FILE:-${DOCKER_IMAGE_NAME:-}}

if [ -z "$DOCKER_IMAGE_DIR" ] || [ -z "$DOCKER_IMAGE_FILE" ] || [ -z "$DOCKER_IMAGE_REF" ]; then
    echo "Error: DOCKER_IMAGE_DIR, DOCKER_IMAGE_FILE (tar basename), and DOCKER_IMAGE_REF are required."
    echo "Usage: $0 <dir> <tar_file> <image_ref>"
    echo "Example: $0 /path/to/dir python_3.11-slim.tar python:3.11-slim"
    echo "Env: DOCKER_IMAGE_DIR, DOCKER_IMAGE_FILE (or legacy DOCKER_IMAGE_NAME), DOCKER_IMAGE_REF"
    exit 1
fi

OUT_PATH="$DOCKER_IMAGE_DIR/$DOCKER_IMAGE_FILE"
DOCKER_INSTALL_METHOD=${DOCKER_INSTALL_METHOD:-docker}
PULL_REF=$(apply_dockerhub_mirror "$DOCKER_IMAGE_REF")

mkdir -p "$DOCKER_IMAGE_DIR" || exit 1

case "$DOCKER_INSTALL_METHOD" in
docker)
    echo "=== docker pull $PULL_REF ==="
    docker pull "$PULL_REF" || exit 1
    echo "=== docker save -> $OUT_PATH ==="
    docker save -o "$OUT_PATH" "$PULL_REF" || exit 1
    ;;
skopeo)
    SKOPEO_BIN="${SKOPEO:-}"
    if [ -z "$SKOPEO_BIN" ] && command -v skopeo >/dev/null 2>&1; then
        SKOPEO_BIN=$(command -v skopeo)
    fi
    if [ -z "$SKOPEO_BIN" ] && [ -x /usr/bin/skopeo ]; then
        SKOPEO_BIN=/usr/bin/skopeo
    fi
    if [ -z "$SKOPEO_BIN" ] || [ ! -x "$SKOPEO_BIN" ]; then
        echo "Error: skopeo not found. Install with: dnf -y install skopeo --nobest"
        echo "       (use --nobest if Docker CE containerd.io is installed), or set SKOPEO=/path/to/skopeo"
        echo "       Current PATH: $PATH"
        exit 1
    fi
    echo "=== skopeo copy docker://$PULL_REF -> docker-archive:$OUT_PATH ==="
    "$SKOPEO_BIN" copy "docker://$PULL_REF" "docker-archive:$OUT_PATH" || exit 1
    ;;
crane)
    if ! command -v crane >/dev/null 2>&1; then
        echo "Error: crane not found. Install crane or use DOCKER_INSTALL_METHOD=docker."
        exit 1
    fi
    echo "=== crane pull $PULL_REF -> $OUT_PATH ==="
    crane pull "$PULL_REF" "$OUT_PATH" || exit 1
    ;;
*)
    echo "Error: unknown DOCKER_INSTALL_METHOD=$DOCKER_INSTALL_METHOD (use docker, skopeo, or crane)"
    exit 1
    ;;
esac

echo "=== Install completed ==="
