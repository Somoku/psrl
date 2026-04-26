DOCKER_IMAGE_TAG="python:3.11-slim"
DOCKER_IMAGE_FILE="python_3.11-slim.tar"

DOCKERHUB_MIRROR=docker.m.daocloud.io \
DOCKER_INSTALL_METHOD=skopeo \
DOCKER_IMAGE_DIR=/jizhicfs/lhy/docker_images \
DOCKER_IMAGE_FILE=$DOCKER_IMAGE_FILE \
DOCKER_IMAGE_TAG=$DOCKER_IMAGE_TAG \
/jizhicfs/lhy/psrl_agent/scripts/docker/docker_install.sh \

DOCKER_NODE_IPS=28.49.196.175:8,28.49.196.77:8,28.58.226.5:8,28.49.38.163:8,29.162.234.163:8,28.49.37.141:8,28.59.83.117:8,29.162.224.113:8 DOCKER_NODE_NUM=8 \
DOCKER_IMAGE_DIR=/jizhicfs/lhy/docker_images DOCKER_IMAGE_FILE=$DOCKER_IMAGE_FILE DOCKER_IMAGE_TAG=$DOCKER_IMAGE_TAG \
/jizhicfs/lhy/psrl_agent/scripts/docker/docker_copy.sh
