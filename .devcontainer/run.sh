#!/bin/bash

# Script to run python code in docker container
#
# NOTE: if `--gpus=all` ever stops working with CUDA error 999 while
# nvidia-smi still works, the CDI spec (/etc/cdi/nvidia.yaml) has a stale
# major number for /dev/nvidia-uvm (it is assigned dynamically at boot).
# Regenerate it with:
#   sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

docker run --tty --rm \
    --gpus=all \
    --shm-size=20g \
    -v /mnt:/mnt \
    -v /tmp/torch-cache:/home/ubuntu/.cache/torch \
    -v .:/app \
    -w /app \
    -e PYTHONPATH="/app:/app/tmp/detr-repo/src:${PYTHONPATH}" \
    -e PYTORCH_CUDA_ALLOC_CONF \
    hmcvlab/computer-vision:latest \
    python3 "$@"
