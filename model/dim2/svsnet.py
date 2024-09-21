import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride =1,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride =1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
      
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
      
        x += residual
        
        return F.relu(x)

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class SpatialDropout3D(nn.Module):
    def __init__(self, p=0.5, data_format="channels_first"):
        super(SpatialDropout3D, self).__init__()
        self.p = p
        self.data_format = data_format

    def forward(self, x):
        if self.data_format == "channels_first":
            x = x.permute(0, 2, 3, 4, 1)  # 转换为 (batch, depth, height, width, channels)
            x = nn.functional.dropout3d(x, self.p, self.training)
            x = x.permute(0, 4, 1, 2, 3)  # 转换回 (batch, channels, depth, height, width)
        else:  # 默认是 "channels_last"
            x = nn.functional.dropout3d(x, self.p, self.training)
        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, iter, depth):
        super(AttentionBlock, self).__init__()
        self.iter = iter
        self.depth = depth
        self.global_pool = nn.AdaptiveMaxPool3d(1)
        self.conv_1x1 = nn.Conv3d(depth, depth, kernel_size=1, stride=1, padding=0)
        self.conv_2x1 = nn.Conv3d(depth, depth, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        global_pool = self.global_pool(x)  # (batch_size, channels, 1, 1, 1)
        conv_1x1 = self.conv_1x1(global_pool)
        relu_out = self.relu(conv_1x1)
        conv_2x1 = self.conv_2x1(relu_out)
        sigmoid_out = self.sigmoid(conv_2x1)
        
        # Repeat the sigmoid_out tensor along each dimension according to the loops
        concat1 = sigmoid_out
        for _ in range(4 - 1):
            concat1 = torch.cat([concat1, sigmoid_out], dim=2)
            
        concat2 = concat1
        for _ in range(self.iter - 1):
            concat2 = torch.cat([concat2, concat1], dim=3)
            
        concat3 = concat2
        for _ in range(self.iter - 1):
            concat3 = torch.cat([concat3, concat2], dim=4)
        
        out = x * concat3  # Element-wise multiplication
        return out



class SaliencyMapAttentionBlock(nn.Module):
    def __init__(self, depth):
        super(SaliencyMapAttentionBlock, self).__init__()
        self.conv1 = nn.Conv3d(depth, depth, kernel_size=1)
        self.conv2 = nn.Conv3d(depth, depth, kernel_size=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = torch.sigmoid(self.conv2(x1))
        return x + x * x2

class ChannelAttentionBlock(nn.Module):
    def __init__(self, depth, size):
        super(ChannelAttentionBlock, self).__init__()
        self.depth = depth
        self.size = size

        # Global Average Pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 1x1 Convolutions
        self.conv_1x1 = nn.Conv2d(2 * depth, depth, kernel_size=1, stride=1, padding=0)
        self.conv_2x1 = nn.Conv2d(depth, depth, kernel_size=1, stride=1, padding=0)

        # Activation layers
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, low_input, high_input):
        # Concatenate low_input and high_input along the channel dimension
        x = torch.cat([low_input, high_input], dim=1)

        # Apply Global Average Pooling
        global_pool = self.global_pool(x)

        # Reshape and apply the 1x1 convolutions with ReLU and Sigmoid activations
        conv_1x1 = self.conv_1x1(global_pool)
        relu_out = self.relu(conv_1x1)
        conv_2x1 = self.conv_2x1(relu_out)
        sigmoid_out = self.sigmoid(conv_2x1)

        # Repeat along height and width dimensions
        concat1 = sigmoid_out.expand(-1, -1, self.size, 1)
        concat2 = concat1.expand(-1, -1, -1, self.size)

        # Element-wise multiplication of low_input and concat2
        out1 = low_input * concat2

        # Element-wise addition of out1 and high_input
        out2 = out1 + high_input

        return out2

class SVSNet(nn.Module):
    def __init__(self, n_channels, frame, patch_height, patch_width):
        super(SVSNet, self).__init__()
        self.conv0 = nn.Conv3d(n_channels, 8, kernel_size=1)

        self.conv1 = ConvBlock3D(8, 8)
        self.conv1_3d_2d = nn.Conv3d(8, 8, kernel_size=(4, 1, 1))
        # self.conv1_trans_2d = nn.Conv2d(8, 8, kernel_size=1)
        
        self.conv1_1 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(1,2,2), stride=(1, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )

        self.conv2 = ConvBlock3D(16, 16)
        self.conv2_3d_2d = nn.Conv3d(16, 16, kernel_size=(4, 1, 1))
        # self.conv2_trans_2d = nn.Conv2d(16, 16, kernel_size=1)
        
        self.conv2_1 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(1,2,2), stride=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.conv3 = ConvBlock3D(32, 32)
        self.conv3_3d_2d = nn.Conv3d(32, 32, kernel_size=(4, 1, 1))
        # self.conv3_trans_2d = nn.Conv2d(32, 32, kernel_size=1)
        
        self.conv3_1 = nn.Sequential(
            nn.Conv3d(32, 64, (1,2,2), stride=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.conv4 = ConvBlock3D(64, 64)
        self.conv4_3d_2d = nn.Conv3d(64, 64, kernel_size=(4, 1, 1))
        # self.conv4_trans_2d = nn.Conv2d(64, 64, kernel_size=1)
        
        self.conv4_1 = nn.Sequential(
            nn.Conv3d(64, 128, (1,2,2), stride=(1, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )

        self.conv5 = ConvBlock3D(128, 128)
        self.conv5_3d_2d = nn.Conv3d(128, 128, kernel_size=(4, 1, 1))
        # self.conv5_trans_2d = nn.Conv2d(128, 128, kernel_size=1)
        
        self.conv5_1 = nn.Sequential(
            nn.Conv3d(128, 256, (1,2,2), stride=(1, 2, 2)),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

        self.conv6 = ConvBlock3D(256, 256)
        self.conv6_3d_2d = nn.Conv3d(256, 256, kernel_size=(4, 1, 1))
        # self.conv6_trans_2d = nn.Conv2d(256, 256, kernel_size=1)

        self.conv6_1 = nn.Sequential(
            nn.Conv3d(256, 512, (1,2,2), stride=(1, 2, 2)),
            nn.BatchNorm3d(512),
            nn.ReLU()
        )

        self.conv7 = ConvBlock3D(512, 512)
        self.conv7_3d_2d = nn.Conv3d(512,512, kernel_size=(4, 1, 1))
        # self.conv6_trans_2d = nn.Conv2d(256, 256, kernel_size=1)

       

        self.drop = SpatialDropout3D(0.5, data_format='channels_first')
        self.atten =AttentionBlock(16,256)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.up1_2 = ChannelAttentionBlock(256,16)
        self.up1_3 = ConvBlock2D(256, 256)

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up2_2 = ChannelAttentionBlock(128,32)
        self.up2_3 = ConvBlock2D(128, 128)

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.up3_2 = ChannelAttentionBlock(64,64)
        self.up3_3 = ConvBlock2D(64, 64)

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up4_2 = ChannelAttentionBlock(32,128)
        self.up4_3 = ConvBlock2D(32, 32)

        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.up5_2 = ChannelAttentionBlock(16,256)
        self.up5_3 = ConvBlock2D(16, 16)

        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.up6_2 = ChannelAttentionBlock(8,512)
        self.up6_3 = ConvBlock2D(8, 8)

        self.out_conv = nn.Conv2d(8, 2, kernel_size=1)

    def forward(self, x):
        x0 = self.conv0(x)
      
        x1 = self.conv1(x0)
      
        x1_2d = self.conv1_3d_2d(x1).squeeze(2)
     
        # x1_2d = self.conv1_trans_2d(x1_2d)
     

        x2 = self.conv1_1(x1)
      
        x2 = self.conv2(x2)
       
        x2_2d = self.conv2_3d_2d(x2).squeeze(2)
       
        # x2_2d = self.conv2_trans_2d(x2_2d)

        x3 = self.conv2_1(x2)
       
        x3 = self.conv3(x3)
      
        x3_2d = self.conv3_3d_2d(x3).squeeze(2)
       
        # x3_2d = self.conv3_trans_2d(x3_2d)

        x4 = self.conv3_1(x3)
       
        x4 = self.conv4(x4)
       
        x4_2d = self.conv4_3d_2d(x4).squeeze(2)
       
        # x4_2d = self.conv4_trans_2d(x4_2d)

        x5 = self.conv4_1(x4)
      
        x5 = self.drop(x5)
        x5 = self.conv5(x5)
        
        x5_2d = self.conv5_3d_2d(x5).squeeze(2)
        
        # x5_2d = self.conv5_trans_2d(x5_2d)

        x6 = self.conv5_1(x5)
        
        x6 = self.atten(x6)
        x6 = self.conv6(x6)
        
        x6_2d = self.conv6_3d_2d(x6).squeeze(2)
        x6 =self.drop(x6)
        
        # x6_2d = self.conv6_trans_2d(x6_2d)
        x7 = self.conv6_1(x6)
       
       
        x7 = self.conv7(x7)
       
        x7_2d = self.conv7_3d_2d(x7).squeeze(2)

        up1 = self.up1(x7_2d)
        up1 = self.up1_3(up1)

        
        up1 = self.up1_2(x6_2d, up1)
        
        
        
        up2 = self.up2(up1)
        up2 = self.up2_3(up2)
        
        up2 = self.up2_2(up2, x5_2d)
        
        up3 = self.up3(up2)
        up3 = self.up3_3(up3)
        
        up3 = self.up3_2(up3, x4_2d)
        

        up4 = self.up4(up3)
        up4 = self.up4_3(up4)
        up4 = self.up4_2(up4, x3_2d)
        

        up5 = self.up5(up4)
        up5 = self.up5_3(up5)
        up5 = self.up5_2(up5, x2_2d)

        up6 = self.up6(up5)
        up6 = self.up6_3(up6)
        up6 = self.up6_2(up6, x1_2d)
        

        out = self.out_conv(up6)

        return out


# Example usage
# n_channels = 1
# frame = 4
# patch_height = 512
# patch_width = 512

# # 初始化模型
# model = SVSNet(n_channels=n_channels, frame=frame, patch_height=patch_height, patch_width=patch_width)
# input_tensor = torch.randn(1, 1, 4, 512, 512)
# output = model(input_tensor)
# # print(output.shape)  # Should output torch.Size([1, 1, 64, 64])
