import time
import torch
from thop import profile

from models.Unet import UNet
from models.Unet3d import UNet3D
from models.Unetr import UNETR
from monai.networks.nets import SegResNet
from monai.networks.nets import SwinUNETR
from models.nnMamba import nnMamba
from models.SegMamba import SegMamba
from models.SwinUNETR import SwinUNETR 
from models.SlimUNETR import SlimUNETR
from models.SlimUNETRv2 import SlimUNETR as SlimUNETRv2

def test_weight(model, device):
    time.sleep(3)
    model = model.to(device)
    model.eval()
    try:
        with torch.inference_mode():
            x = torch.zeros(size=(1, 4, 128, 128, 128)).to(device)
            torch.cuda.synchronize()
            for i in range(0, 3):
                _ = model(x)
                torch.cuda.synchronize()
            x = torch.randn(size=(1, 4, 128, 128, 128)).to(device)
            start_time = time.time()
            torch.cuda.synchronize()
            for i in range(0, 3):
                _ = model(x)
                torch.cuda.synchronize()
            end_time = time.time()
            need_time = end_time - start_time
            # print(need_time)
            flops, params = profile(model, inputs=(x,))
            throughout = round( x.shape[0] / (need_time / 3), 3)
            return flops, params, throughout
    except torch.cuda.OutOfMemoryError as e:
        print(f'{e}')
        return 0,0,0



def Unitconversion(flops, params, throughout):
    print('params : {} M'.format(round(params / (1000**2), 2)))
    print('flop : {} G'.format(round(flops / (1000**3), 2)))
    print('throughout: {} FPS'.format(throughout))

def inference_times(name, model, device):
    print(f"################### {name} in {torch.cuda.get_device_name()} ###################")
    flops, param, throughout = test_weight(model, device)
    if flops!=0 and param!=0 and throughout!=0:
        Unitconversion(flops, param, throughout)

if __name__ == "__main__":
    device = 'cuda:0'
    
    # U-Net
    name = "U-Net"
    # model = UNet(in_channels=4, num_classes=3)
    model = UNet3D()
    inference_times(name, model, device)
    # UNETR
    name = "UNETR"
    model = UNETR(in_channels=4,out_channels=3,img_size=(128, 128, 128),feature_size=16,hidden_size=768,mlp_dim=3072,num_heads=12,pos_embed="perceptron",norm_name="instance",conv_block=True,res_block=True,dropout_rate=0.0,)
    inference_times(name, model, device)
    # SegResNet
    name = "SegResNet"
    model = SegResNet(spatial_dims=3, init_filters=24, in_channels=4, out_channels=3, blocks_up=(4, 4, 4))
    inference_times(name, model, device)
    # Swin UNETR
    name = "Swin UNETR"
    model = SwinUNETR(in_channels=4 ,out_channels=3, img_size=128, feature_size=48)
    inference_times(name, model, device)
    # nnMamba
    name = "nnMamba"
    model = nnMamba(in_ch=4, number_classes=3)
    inference_times(name, model, device)
    # SegMamba
    name = "SegMamba"
    model = SegMamba(in_chans=4, out_chans=3)
    inference_times(name, model, device)
    # SlimUNETR
    name = "SlimUNETR"
    model = SlimUNETR(in_channels=4, out_channels=3, embed_dim=96,embedding_dim=64, channels=(24, 48, 60),
                        blocks=(2, 2, 4, 4), heads=(1, 2, 4, 4), r=(4, 2, 2, 1), distillation=False,
                        dropout=0.3)
    inference_times(name, model, device)
    # SlimUNETRv2
    name = "SlimUNETR v2"
    model = SlimUNETRv2(in_chans=4, out_chans=3, kernel_sizes=[4, 2, 2, 1], depths=[1, 1, 1, 1], dims=[48, 96, 192, 384], heads=[1, 2, 2, 2], hidden_size=768, num_slices_list = [64, 32, 16, 8],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3])
    inference_times(name, model, device)