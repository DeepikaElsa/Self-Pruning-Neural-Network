# test_gpu.py
import torch
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"GPU name        : {torch.cuda.get_device_name(0)}")