import torch
import torch.nn as nn

# from dim2.svsnet import SVSNet
#
# model = SVSNet(1, 4,512,512)

# from dim2 import AttentionUNet
#
# model = AttentionUNet(1, 2, 32)

# from dim2 import SwinUnet
# from dim2.swin_unet import SwinUnet_config
# config = SwinUnet_config()
# model = SwinUnet(config, img_size=512, num_classes=2)

# from dim2.Angionet import AngioNet # type: ignore
# model = AngioNet()
from dim2.multi_DEGCN import DEDCGCNEE
model = DEDCGCNEE(32, 2,5)
# total_params = sum(p.numel() for p in model.parameters())
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params_in_millions = total_params / 1_000_000

print(f"Total number of parameters: {total_params_in_millions:.2f} million")
