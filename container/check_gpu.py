import os


os.environ["DGLBACKEND"] = "pytorch"

import torch

import dgl

from dgl.data import DGLDataset


# Check if CUDA is available
is_cuda_available = torch.cuda.is_available()

print("Is CUDA available:", is_cuda_available)
# If CUDA is available, it will print the number of available GPUs and their names
if is_cuda_available:
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
print(f"torch vesion: {torch.__version__}")
print("dgl version:",dgl.__version__)

