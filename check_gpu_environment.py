import sys
import platform
import importlib.metadata as metadata

print("===== OPENVLA GPU ENVIRONMENT CHECK =====")
print("Python:", sys.version.split()[0])
print("Operating system:", platform.platform())

for package in ("torch", "transformers", "accelerate", "safetensors", "Pillow"):
    try:
        print(f"{package}: {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")

try:
    import torch
except ImportError:
    print("PyTorch unavailable")
    raise SystemExit(1)

print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("\nSTATUS: GPU NOT AVAILABLE")
    print("CPU-side preparation can continue.")
    print("Real OpenVLA inference has not been tested.")
    raise SystemExit(0)

for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    total_gb = properties.total_memory / (1024 ** 3)

    print(f"\nGPU {index}: {properties.name}")
    print(f"Total VRAM: {total_gb:.2f} GiB")
    print(
        "Compute capability:",
        f"{properties.major}.{properties.minor}",
    )

    if properties.major >= 8:
        print("BF16 hardware support: likely")
    else:
        print("BF16 hardware support: requires verification")

    if total_gb < 16:
        print("WARNING: Memory likely insufficient for standard OpenVLA-7B loading")
    elif total_gb < 24:
        print("WARNING: Memory headroom may be limited")
    else:
        print("GPU memory capacity: promising; loading still needs testing")

print("\nSTATUS: GPU DETECTED")
print("Next step: validate actual OpenVLA model loading.")
