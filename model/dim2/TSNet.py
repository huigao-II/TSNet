

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 23:05:23 2020

@author: zhang
"""
import cv2
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np
from matplotlib import pyplot as plt
from .deseatten import TemporalAttention
from .sobel import edge_conv2d64
from .sobel import edge_conv2d128
from .sobel import edge_conv2d256,pre_edge
from .CBAM import CBAM
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = residual + out
        out = self.relu(out)
        return out
class Encoder(nn.Module):
    def __init__(self, in_c):
        super(Encoder, self).__init__()

        self.down = downsample()

        self.Conv1 = conv_block(ch_in=in_c, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.down(x1)
        x2 = self.Conv2(x2)

        x3 = self.down(x2)
        x3 = self.Conv3(x3)

        x4 = self.down(x3)
        x4 = self.Conv4(x4)
        return x4
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out    

def downsample():
    return nn.MaxPool2d(kernel_size=2, stride=2)

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class GCN(nn.Module):
    def __init__(self, channel):
        super(GCN, self).__init__()  
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.para = torch.nn.Parameter(torch.ones((1,512*3,64,64), dtype = torch.float32))
        self.adj = torch.nn.Parameter(torch.ones((512*3,512*3), dtype = torch.float32))
             
    def forward(self, x):
        # y = torch.nn.functional.relu(self.adj)
        b, c, H, W = x.size()
        fea_matrix = x.view(b,c,H*W)
        c_adj = self.avg_pool(x).view(b,c)

        m = torch.ones((b,c,H,W), dtype = torch.float32)

        for i in range(0,b):

         t1 = c_adj[i].unsqueeze(0)
         t2 = t1.t()
         c_adj_s = torch.abs(torch.abs(torch.sigmoid(t1-t2)-0.5)-0.5)*2
         c_adj_s = (c_adj_s.t() + c_adj_s)/2

         output0 = torch.mul(torch.mm(self.adj*c_adj_s,fea_matrix[i]).view(1,c,H,W),self.para)

         m[i] = output0

        output = torch.nn.functional.relu(m.cuda())

        return output

class EEblock(nn.Module):
    def __init__(self, channel):
        super(EEblock, self).__init__()  
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.sconv13 = nn.Conv2d(channel,channel, kernel_size=(1,3), padding=(0,1))
        self.sconv31 = nn.Conv2d(channel,channel, kernel_size=(3,1), padding=(1,0))
             
    def forward(self, y, x):
        # y = torch.nn.functional.relu(self.adj)
        b, c, H, W = x.size()

        x1 = self.sconv13(x)
        x2 = self.sconv31(x)

        y1 = self.sconv13(y)
        y2 = self.sconv31(y)

        map_y13 = torch.sigmoid(self.avg_pool(y1).view(b,c,1,1))
        map_y31 = torch.sigmoid(self.avg_pool(y2).view(b,c,1,1))

        k = x1*map_y31 + x2*map_y13 + x

        return k



class DEDCGCNEE(nn.Module):
    def __init__(self, in_c, n_classes,frameN):
        super(DEDCGCNEE, self).__init__()
        self.frameN = frameN
        self.down = downsample()

        self.Conv1 = conv_block(ch_in=in_c, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.n_classes = n_classes
        self.Conv5 = conv_block(ch_in=in_c, ch_out=64)
        self.Conv6 = conv_block(ch_in=64, ch_out=128)
        self.Conv7 = conv_block(ch_in=128, ch_out=256)
        self.Conv8 = conv_block(ch_in=256, ch_out=512)
        # self.get_atten = TemporalAttention()
        self.get_atten = CBAM(in_planes=512*frameN)
        self.GCN_layer = GCN(channel=512)

        self.EEblock1 = EEblock(channel=256)
        self.EEblock2 = EEblock(channel=128)
        self.EEblock3 = EEblock(channel=64)

        self.Up4 = up_conv(512,256)
        self.Up_conv4 = Decoder(512, 256)

        self.Up3 = up_conv(256,128)
        self.Up_conv3 = Decoder(256, 128)

        self.Up2 = up_conv(128,64)
        self.Up_conv2 = Decoder(128, 64)

        self.fconv = nn.Conv2d(64, self.n_classes, kernel_size=1, padding=0)

    def forward(self,x_stack):
        x4out = torch.zeros(x_stack.shape[0],512,64,64).cuda()
        x4out_list = []
        for i in range(x_stack.shape[2]):
            x = x_stack[:,:,i,:,:]   # 将stack
            # 可视化
        #     for m in range(x.shape[0]):
        #         show = x.squeeze(1)[m]
        #         plt.subplot(3,4,4*i+m+1)
        #         plt.imshow(show.cpu(),cmap="gray")
        # plt.show()

            x1 = self.Conv1(x)

            x2 = self.down(x1)
            x2 = self.Conv2(x2)

            x3 = self.down(x2)
            x3 = self.Conv3(x3)

            x4 = self.down(x3)
            x4 = self.Conv4(x4)

            if i == self.frameN//2:
                y = pre_edge(x)
                x1_=x1
                x2_=x2
                x3_=x3
                x4_=x4
                e1 = edge_conv2d64(x1)
                e2 = edge_conv2d128(x2)
                e3 = edge_conv2d256(x3)

                y1 = self.Conv5(y) + e1

                y2 = self.down(y1)
                y2 = self.Conv6(y2) + e2

                y3 = self.down(y2)
                y3 = self.Conv7(y3) + e3

                y4 = self.down(y3)
                y4 = self.Conv8(y4)
                x4 = x4+y4

            x4out_list.append(x4)




        # showxout = xout.squeeze(1)[0][0]
        # plt.subplot(1,2,1)
        # plt.imshow(showxout.cpu(),cmap="gray")
        # showx4 = x4_.squeeze(1)[0][0]
        # plt.subplot(1, 2, 2)
        # plt.imshow(showx4.cpu(), cmap="gray")
        # plt.show()

        # B,T,C,H,W = x4stack.shape
        # x4stack = x4stack.reshape(B, T * C, H, W)

        x4stack = torch.stack(x4out_list).permute(1, 0, 2, 3, 4)  # B,T,C,H,W
        atten_out = self.get_atten(x4stack)
        ########################可视化

        #####################################
        x4_out = atten_out
        # GCN_output = self.GCN_layer(x4stack + y4)
        # GCN_output = self.GCN_layer(x4stack)
        # GCN_output = GCN_output.reshape(B, T, C, H, W)
        # GCN_output = torch.sum(GCN_output,1)

        # GCN_output = self.GCN_layer(x4_out+y4)
        #
        # d4 = self.Up4(GCN_output)

        d4 = self.Up4(x4_out+x4_)
        m3 = self.EEblock1(y3,x3_)
        l4 = torch.cat((m3, d4), dim=1)
        d4 = self.Up_conv4(l4)

        d3 = self.Up3(d4)
        m2 = self.EEblock2(y2,x2_)
        l3 = torch.cat((m2, d3), dim=1)
        d3 = self.Up_conv3(l3)

        d2 = self.Up2(d3)
        m1 = self.EEblock3(y1,x1_)
        l2 = torch.cat((m1, d2), dim=1)
        d2 = self.Up_conv2(l2)

        out = self.fconv(d2)
        return out

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.Tensor(4, 1, 3, 512, 512).cuda()
    x.to(device)
    print("x size: {}".format(x.size()))

    model = DEDCGCNEE(1, 2, 3)
    model = model.cuda()
    out = model(x)
    print("out size: {}".format(out.size()))
