# inspect_model.py

import torch

path = "results/CA-HepTh/subset_0_10/watermarked_model.pth"

state_dict = torch.load(path, map_location="cpu")
print("🔍 Keys in state_dict:")
for key in state_dict.keys():
    print("  -", key)
