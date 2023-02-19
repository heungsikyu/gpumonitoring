import torch
import platform

import PIL
print(PIL.__version__)

print (f"PyTorch version:{torch.__version__}") # 1.12.1 이상
print(f"MPS 장치를 지원하도록 build 되었는지: {torch.backends.mps.is_built()}") # True 여야 합니다.
print(f"MPS 장치가 사용 가능한지: {torch.backends.mps.is_available()}") # True 여야 합니다.
print(f"MPS 장치가 설치 되어 있는지: {torch.backends.mps.is_built()}")
print(torch.cuda.is_available())
print(platform.platform())

# import torch

# GPU가 사용 가능한 경우 CUDA 디바이스를 사용합니다.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu",

# 두 개의 Tensor를 생성합니다.
x = torch.randn(10, 10)
y = torch.randn(10, 10) 

# # Tensor를 GPU로 옮깁니다.
# x = x.to(device) 
# y = y.to(device) 

# GPU에서 계산을 수행합니다.
z = x + y

# 결과를 다시 CPU로 옮깁니다.
z = z.to("cpu")

print(z)