import torch 
import torch.nn as nn

class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock2d, self).__init__()

        # 第1个3*3的卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # 第2个3*3的卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    # 定义数据前向流动形式
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


is_elu = False
def activateELU(is_elu, nchan):
    if is_elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

def ConvBnActivate(in_channels, middle_channels, out_channels):
    # This is a block with 2 convolutions
    # The first convolution goes from in_channels to middle_channels feature maps
    # The second convolution goes from middle_channels to out_channels feature maps
    conv = nn.Sequential(
        nn.Conv3d(in_channels, middle_channels, stride=1, kernel_size=3, padding=1),
        nn.BatchNorm3d(middle_channels),
        activateELU(is_elu, middle_channels),

        nn.Conv3d(middle_channels, out_channels, stride=1, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        activateELU(is_elu, out_channels),
    )
    return conv


def DownSample():
    # It halves the spatial dimensions on every axes (x,y,z)
    return nn.MaxPool3d(kernel_size=2, stride=2)

def UpSample(in_channels, out_channels):
    # It doubles the spatial dimensions on every axes (x,y,z)
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

def FinalConvolution(in_channels, out_channels):
    return nn.Conv3d(in_channels, out_channels, kernel_size=1)


def CatBlock(x1, x2):
    return torch.cat((x1, x2), 1)


class UNet3D(nn.Module):
    def __init__(self, num_out_classes=3, input_channels=4, init_feat_channels=10, testing=False):
        super().__init__()

        self.testing = testing

        # Encoder layers definitions
        self.down_sample = DownSample()

        self.init_conv = ConvBnActivate(input_channels, init_feat_channels, init_feat_channels*2)
        self.down_conv1 = ConvBnActivate(init_feat_channels*2, init_feat_channels*2, init_feat_channels*4)
        self.down_conv2 = ConvBnActivate(init_feat_channels*4, init_feat_channels*4, init_feat_channels*8)
        self.down_conv3 = ConvBnActivate(init_feat_channels*8, init_feat_channels*8, init_feat_channels*16)

        # Decoder layers definitions
        self.up_sample1 = UpSample(init_feat_channels*16, init_feat_channels*16)
        self.up_conv1   = ConvBnActivate(init_feat_channels*(16+8), init_feat_channels*8, init_feat_channels*8)

        self.up_sample2 = UpSample(init_feat_channels*8, init_feat_channels*8)
        self.up_conv2   = ConvBnActivate(init_feat_channels*(8+4), init_feat_channels*4, init_feat_channels*4)

        self.up_sample3 = UpSample(init_feat_channels*4, init_feat_channels*4)
        self.up_conv3   = ConvBnActivate(init_feat_channels*(4+2), init_feat_channels*2, init_feat_channels*2)

        self.final_conv = FinalConvolution(init_feat_channels*2, num_out_classes)

        # Softmax
        self.softmax = nn.Softmax(dim=1)    # 多分类问题用soft-max函数作为输出
        self.sigmoid = nn.Sigmoid()         # 二分类问题用sigmoid函数作为输出，二分类下和softmax等价

    def forward(self, image):
        # Encoder Part #
        # B x  1 x Z x Y x X
        layer_init = self.init_conv(image)

        # B x 64 x Z x Y x X
        max_pool1  = self.down_sample(layer_init)
        # B x 64 x Z//2 x Y//2 x X//2
        layer_down2 = self.down_conv1(max_pool1)

        # B x 128 x Z//2 x Y//2 x X//2
        max_pool2   = self.down_sample(layer_down2)
        # B x 128 x Z//4 x Y//4 x X//4
        layer_down3 = self.down_conv2(max_pool2)

        # B x 256 x Z//4 x Y//4 x X//4
        max_pool_3  = self.down_sample(layer_down3)
        # B x 256 x Z//8 x Y//8 x X//8
        layer_down4 = self.down_conv3(max_pool_3)
        # B x 512 x Z//8 x Y//8 x X//8

        # Decoder part #
        layer_up1 = self.up_sample1(layer_down4)
        # B x 512 x Z//4 x Y//4 x X//4
        cat_block1 = CatBlock(layer_down3, layer_up1)
        # B x (256+512) x Z//4 x Y//4 x X//4
        layer_conv_up1 = self.up_conv1(cat_block1)
        # B x 256 x Z//4 x Y//4 x X//4

        layer_up2 = self.up_sample2(layer_conv_up1)
        # B x 256 x Z//2 x Y//2 x X//2
        cat_block2 = CatBlock(layer_down2, layer_up2)
        # B x (128+256) x Z//2 x Y//2 x X//2
        layer_conv_up2 = self.up_conv2(cat_block2)
        # B x 128 x Z//2 x Y//2 x X//2

        layer_up3 = self.up_sample3(layer_conv_up2)
        # B x 128 x Z x Y x X
        cat_block3 = CatBlock(layer_init, layer_up3)
        # B x (64+128) x Z x Y x X
        layer_conv_up3 = self.up_conv3(cat_block3)

        # B x 64 x Z x Y x X
        final_layer = self.final_conv(layer_conv_up3)
        # B x 2 x Z x Y x X
        if self.testing:
            final_layer = self.sigmoid(final_layer)

        return final_layer

if __name__ == '__main__':
    x = torch.randn(1, 4, 128, 128, 128)
    net = UNet3D()
    y = net(x)
    # print("params: ", sum(p.numel() for p in net.parameters()))
    print(y.shape)