#!/usr/bin/env bash

set -exu

IMAGE_TAG='pkm-vim:latest'

# Figure out repository root.
SCRIPT_DIR="$( cd "$( dirname $(realpath "${BASH_SOURCE[0]}") )" && pwd )"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"


docker run -it \
    --mount type=bind,source=${REPO_ROOT},target="/home/user/$(basename ${REPO_ROOT})" \
    --shm-size=8g \
    --network host \
    --privileged \
    --gpus all \
    "$@" \
    "${IMAGE_TAG}"
