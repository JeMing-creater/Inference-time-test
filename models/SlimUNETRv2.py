# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch.nn as nn
import torch 
import time
from functools import partial

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F 

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i)))
    
# Swish激活函数
class Swish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # bimamba_type="v3",
                # nslices=num_slices,
        )
    
    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip
        
        return out
    
class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, shallow=True):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        if shallow == True:
            self.act = nn.GELU()
        else:
            self.act = Swish()
        # self.act = nn.ReLU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GSC(nn.Module):
    def __init__(self, in_channles, shallow=True) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        if shallow == True:
            self.nonliner = nn.GELU()
        else:
            self.nonliner = Swish()
        # self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        if shallow == True:
            self.nonliner2 = nn.GELU()
        else:
            self.nonliner2 = Swish()
        # self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        if shallow == True:
            self.nonliner3 = nn.GELU()
        else:
            self.nonliner3 = Swish()
        # self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        if shallow == True:
            self.nonliner4 = nn.GELU()
        else:
            self.nonliner4 = Swish()
        # self.nonliner4 = nn.ReLU()

    def forward(self, x):

        x_residual = x 

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        
        return x + x_residual


class SlimMambaEncoder(nn.Module):
    def __init__(self, in_chans=4, kernel_sizes=[4, 2, 2, 2], depths=[2, 2, 2, 2], dims=[48, 96, 192, 384], num_slices_list = [64, 32, 16, 8],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
              nn.Conv3d(in_chans, dims[0], kernel_size=kernel_sizes[0], stride=kernel_sizes[0]),
              )
        
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=kernel_sizes[i+1], stride=kernel_sizes[i+1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        cur = 0
        for i in range(4):
            shallow = True
            if i > 1:
                shallow = False
            gsc = GSC(dims[i], shallow)

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            if i_layer>=2:
                self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer], False))
            else:
                self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer], True))
        
    def forward(self, x):
        # outs = []
        feature_out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            
            feature_out.append(self.stages[i](x))

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x = norm_layer(x)
                x = self.mlps[i](x)
                # outs.append(x_out)   
        return x, feature_out

class TransposedConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, head, r):
        super(TransposedConvLayer, self).__init__()
        self.transposed1 = nn.ConvTranspose3d(dim_in,
                                             dim_out,
                                             kernel_size=r,
                                             stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)
        self.Atten = AttentionLayer(dim_out*2, head, r)
        self.transposed2 = nn.ConvTranspose3d(dim_out*2,
                                             dim_out,
                                             kernel_size=1,
                                             stride=1)

    def forward(self, x, feature):
        x = self.transposed1(x)
        x = torch.cat((x, feature), dim=1)
        x = self.Atten(x)
        x = self.transposed2(x)
        x = self.norm(x)
        return x

class GlobalSparseTransformer(nn.Module):
    def __init__(self, channels, r, heads):
        super(GlobalSparseTransformer, self).__init__()
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5
        self.num_heads = heads
        self.sparse_sampler = nn.AvgPool3d(kernel_size=1, stride=r)
        
        # qkv
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.sparse_sampler(x)
        B, C, H, W, Z = x.shape
        q, k, v = self.qkv(x).view(B, self.num_heads, -1, H * W * Z).split(
            [self.head_dim, self.head_dim, self.head_dim],
            dim=2)
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W, Z)
        return x


class LocalReverseDiffusion(nn.Module):
    def __init__(self, channels, r):
        super(LocalReverseDiffusion, self).__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.conv_trans = nn.ConvTranspose3d(channels,
                                             channels,
                                             kernel_size=r,
                                             stride=r,
                                             groups=channels)
        self.pointwise_conv = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv_trans(x)
        x = self.norm(x)
        x = self.pointwise_conv(x)
        return x

class AttentionLayer(nn.Module):
    def __init__(self, channels, heads, r=2, ):
        super().__init__()
        # self.bn = nn.BatchNorm3d(channels)
        self.GlobalST = GlobalSparseTransformer(channels, r, heads)
        self.LocalRD = LocalReverseDiffusion(channels, r)
        
    def forward(self, x):
        x = self.LocalRD(self.GlobalST(x)) + x
        return x

class SlimUNETR(nn.Module):
    def __init__(self, in_chans=4, out_chans=3, kernel_sizes=[4, 2, 2, 2], depths=[2, 2, 2, 2], dims=[48, 96, 192, 384], heads=[1, 2, 4, 4], hidden_size=768, num_slices_list = [64, 32, 16, 8],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super(SlimUNETR, self).__init__()
        self.Encoder = SlimMambaEncoder(in_chans=in_chans, kernel_sizes=kernel_sizes, depths=depths, dims=dims, num_slices_list = num_slices_list,
                 drop_path_rate=drop_path_rate, layer_scale_init_value=layer_scale_init_value, out_indices=out_indices)

        self.hidden_downsample = nn.Conv3d(dims[3], hidden_size, kernel_size=2, stride=2)
        
        self.TSconv1 = TransposedConvLayer(dim_in=hidden_size, dim_out=dims[3], head=heads[3], r=2)
        
        self.TSconv2 = TransposedConvLayer(dim_in=dims[3], dim_out=dims[2], head=heads[2], r=kernel_sizes[3])
        self.TSconv3 = TransposedConvLayer(dim_in=dims[2], dim_out=dims[1], head=heads[1], r=kernel_sizes[2])
        self.TSconv4 = TransposedConvLayer(dim_in=dims[1], dim_out=dims[0], head=heads[0], r=kernel_sizes[1])

        self.SegHead = nn.ConvTranspose3d(dims[0],out_chans,kernel_size=kernel_sizes[0],stride=kernel_sizes[0])
        
    def forward(self, x):
        outs, feature_out = self.Encoder(x)
        
        deep_feature = self.hidden_downsample(outs)
        
        x = self.TSconv1(deep_feature, feature_out[-1])
        x = self.TSconv2(x, feature_out[-2])
        x = self.TSconv3(x, feature_out[-3])
        x = self.TSconv4(x, feature_out[-4])
        x = self.SegHead(x)
        
        return x
    
def test_weight(model, x):
    for i in range(0, 3):
        _ = model(x)
    start_time = time.time()
    output = model(x)
    end_time = time.time()
    need_time = end_time - start_time
    print(need_time)
    from thop import profile
    flops, params = profile(model, inputs=(x,))
    throughout = round(x.shape[0] / (need_time / 1), 3)
    return flops, params, throughout


def Unitconversion(flops, params, throughout):
    print('params : {} M'.format(round(params / (1000**2), 2)))
    print('flop : {} G'.format(round(flops / (1000**3), 2)))
    print('throughout: {} FPS'.format(throughout))

if __name__ == '__main__':
    device = 'cuda:0'
    # x = torch.randn(size=(1, 4, 96, 96, 96)).to(device)
    x = torch.randn(size=(1, 4, 128, 128, 128)).to(device)

    # model = SegMamba(in_chans=4,out_chans=3).to(device)
    
    model = SlimUNETR(in_chans=4, out_chans=3, kernel_sizes=[4, 2, 2, 2], depths=[2, 2, 2, 2], dims=[48, 96, 192, 384], heads=[1, 2, 4, 4], hidden_size=768, num_slices_list = [64, 32, 16, 8],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]).to(device)
    print(model(x).shape)
    flops, param, throughout = test_weight(model, x)
    Unitconversion(flops, param, throughout)