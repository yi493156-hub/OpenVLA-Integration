# OpenVLA-7B GPU Deployment

## Current status
CPU software tests passed on Intel Mac.
Real OpenVLA inference has not been tested.
Robot action mapping is not calibrated.

## Requirements
- Linux NVIDIA GPU workstation
- Compatible NVIDIA driver and CUDA-enabled PyTorch
- Python 3.10 environment
- Sufficient GPU memory for OpenVLA-7B
- Internet access for model weights on first run

## Setup on the GPU workstation
1. Transfer or clone the project repository.
2. Create a separate Python 3.10 environment.
3. Install a CUDA-compatible PyTorch build for the workstation.
4. Install the official OpenVLA minimal inference requirements:
   python -m pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
5. Check package compatibility, especially transformers, timm and tokenizers.
6. Run:
   bash vla/openvla_integration/launch_real_openvla.sh

## Connect from the Mac
Use an SSH tunnel from the Mac to the GPU workstation:

ssh -N -L 8765:127.0.0.1:8765 USER@GPU_HOST

Replace USER and GPU_HOST with actual values.
Do not expose the unauthenticated inference HTTP service publicly.

Then run the existing evaluation client against:
http://127.0.0.1:8765/predict

## Important limitations
- A running model does not prove robot task success.
- bridge_orig action normalization is for the Bridge dataset.
- The 7D output must not directly control the ORCA robot.
- The arm controller, coordinate transforms and safety limits remain unverified.
- Model API compatibility and GPU memory must be tested on the actual workstation.

Official resources:
https://github.com/openvla/openvla
https://huggingface.co/openvla/openvla-7b
