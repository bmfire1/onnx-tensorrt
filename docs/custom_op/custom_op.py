# Author: boo

import math
import torch
from torch import nn
import torch.nn.functional as F
import json

class MPad2d(nn.Module):
    def __init__(self, value = 0):
        super(MPad2d, self).__init__()
        self.value = value
        
    def forward(self, x, paddings):
        padding = paddings
        if isinstance(paddings,int):
            padding = [paddings] * 4
        elif len(paddings) == 1:
            padding = [paddings[0]] * 4
        elif len(paddings) > 1 and len(paddings) < 4:
            padding = paddings + [0]*(4-len(paddings))
        elif len(paddings) > 4:
            padding = paddings[:4]
        print(padding)
        pad = nn.ConstantPad2d(padding, self.value)        
        x = pad(x)
        return x

#######################upsample
class MupsampleImpl(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, scale_factor, mode, align_corners):
        return g.op("Plugin", input, name_s="Mresize", version_s="1", namespace_s="cust_op",
                     mode_s=mode,
                     coordinate_transformation_mode_s="asymmetric", nearest_mode_s="floor", align_corners_s=str(align_corners),
                     info_s=json.dumps({
                         "scale_factor": str(scale_factor),
                         }),
                     )
    @staticmethod
    def forward(ctx, input1, scale_factor, mode, align_corners):
        if mode=="bilinear":
            align_corners = True
            i = F.interpolate(input1, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
        else:
            i = F.interpolate(input1, scale_factor=scale_factor, mode=mode)
        return i

class MUpsample(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(MUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        if mode=="bilinear":
            self.align_corners = True
    def forward(self, input1):
        return MupsampleImpl.apply(input1, self.scale_factor, self.mode, self.align_corners)
 
#######################custom padding
class MpaddingImpl(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, padding):
        return g.op("Plugin", input, name_s="Mpadding", version_s="1", namespace_s="cust_op", mode_s="constant",
                    info_s=json.dumps({
                        "padding": str(padding),
                        "other": "Hello Onnx Plugin"
                        }),
                    )

    @staticmethod
    def forward(ctx, i, padding):
        i = F.pad(i, padding)
        return i

class Mpadding(nn.Module):
    def __init__(self, padding):
        super(Mpadding, self).__init__()
        self.padding = padding
    def forward(self, x):
        return MpaddingImpl.apply(x, self.padding)

    
'''
实际使用示例
'''

class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        stride_t = stride
        if isinstance(stride, list):
            if len(stride)==1:
                stride_t = stride[0]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride_t,
                              bias=bias, groups=groups)

        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2
        
        if isinstance(stride, list):
            if len(stride)==1:
                self.stride = [stride[0]]*2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2


        p = self.kernel_size[0]- self.stride[0]        
        self.pad = Mpadding(padding=(p//2, p-p//2,p//2,p-p//2))

    def forward(self, x):
        h, w = x.shape[-2:]

        x = self.pad(x) 
        x = self.conv(x)

        return x


