import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50

# 预处理网络
class PreprocessingNetwork(nn.Module):
    def __init__(self):
        super(PreprocessingNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.conv3 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2, dilation=1, bias=False)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2, dilation=1, bias=False)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2, dilation=1, bias=False)
        self.conv6 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.tanh(self.conv6(x))
        x = torch.cat([x, x, x], dim=1)  # 拼接生成3通道的输出
        return x

# AngioNet模型
class AngioNet(nn.Module):
    def __init__(self, L1=0., L2=0.):
        super(AngioNet, self).__init__()
        self.preprocessing_network = PreprocessingNetwork()
        self.deeplab_model = deeplabv3_resnet50(pretrained=False)  # 不加载预训练权重
        self.deeplab_model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.preprocessing_network(x)
        x = self.deeplab_model(x)['out']
        return x


# 使用示例：
# model = AngioNet(L1=0., L2=0., DL_weights=True)  # DL_weights=True将加载预训练权重
# model.eval()

# # 使用随机输入进行测试
# input_tensor = torch.randn(1, 1, 512, 512)  # 批量大小1，单通道，512x512图像
# output = model(input_tensor)
# print(output.shape)



# import torch
# import torch.nn as nn
# import torchvision.models as models

# class UnsharpMaskNet(nn.Module):
#     def __init__(self):
#         super(UnsharpMaskNet, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding=2, bias=False)
#         self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
#         self.conv3 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2, bias=False)
#         self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2, bias=False)
#         self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2, bias=False)
#         self.conv6 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1, bias=False)
#         self.tanh = nn.Tanh()
    
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x2)
#         x4 = self.conv4(x3)
#         x5 = self.conv5(x4)
#         x6 = self.conv6(x5)
#         x6_tanh = self.tanh(x6)
#         x7 = torch.cat([x6_tanh, x6_tanh, x6_tanh], dim=1)
#         return x7

# class Deeplabv3(nn.Module):
#     def __init__(self, weights=None, backbone='xception', num_classes=2):
#         super(Deeplabv3, self).__init__()
#         # Load the pre-trained DeepLabV3 model (using Xception backbone)
#         if backbone == 'xception':
#             self.deeplab = models.segmentation.deeplabv3_xception(pretrained=weights is None)
#         else:
#             raise ValueError('Unsupported backbone')
        
#         # Modify classifier to match num_classes
#         self.deeplab.classifier[4] = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
    
#     def forward(self, x):
#         return self.deeplab(x)['out']

# class AngioNet(nn.Module):
#     def __init__(self, L1=0., L2=0., DL_weights=None):
#         super(AngioNet, self).__init__()
#         self.unsharp_mask_net = UnsharpMaskNet()
#         self.deeplab_net = Deeplabv3(weights=DL_weights, backbone='xception', num_classes=2)
        
#         # Applying L1 and L2 regularization
#         self.l1 = L1
#         self.l2 = L2
    
#     def forward(self, x):
#         x = self.unsharp_mask_net(x)
#         x = self.deeplab_net(x)
#         return x

#     def regularize(self, param):
#         l1_reg = self.l1 * torch.sum(torch.abs(param))
#         l2_reg = self.l2 * torch.sum(param ** 2)
#         return l1_reg + l2_reg

# # Example usage
# model = AngioNet(L1=0.01, L2=0.01)

# input_tensor = torch.randn(1, 1, 512, 512)  # 批量大小1，单通道，512x512图像
# output = model(input_tensor)
# print(output.shape)