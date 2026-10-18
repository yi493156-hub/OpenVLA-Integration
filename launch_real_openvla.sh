#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "===== OPENVLA REAL GPU LAUNCH ====="

python - <<PY
import sys
import torch
import transformers

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: NVIDIA CUDA GPU unavailable. "
        "Real OpenVLA server not started."
    )

device = torch.cuda.get_device_properties(0)
print("GPU:", device.name)
print("VRAM GiB:", round(device.total_memory / 1024**3, 2))

if not torch.cuda.is_bf16_supported():
    raise SystemExit(
        "ERROR: This launcher requires BF16 support. "
        "Choose a compatible precision before deployment."
    )

print("GPU preflight passed")
PY

echo "Starting real OpenVLA inference server..."
python inference_server.py --backend real --host 127.0.0.1 --port 8765
