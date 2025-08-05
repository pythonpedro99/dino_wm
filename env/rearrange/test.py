import torch
ckpt = torch.load("/Users/julianquast/Downloads/outputs/dino/checkpoints/model_latest.pth", map_location="cpu")
print(list(ckpt.keys()))
