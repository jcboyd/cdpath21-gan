import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from math import sqrt
from skimage.io import imsave


class PixelNorm(nn.Module):

    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class EqualConv2d(nn.Module):

    ''' 
        For a more explicit implementation, see:
        https://github.com/rosinality/progressive-gan-pytorch
    ''' 

    def __init__(self, *args, **kwargs):
        super(EqualConv2d, self).__init__()

        self.conv = nn.Conv2d(*args, **kwargs)

        # initialise weights
        self.conv.weight.data.normal_()
        self.conv.bias.data.zero_()

        # compute learning rate multiplier
        fan_in = self.conv.weight.data.size(1) * self.conv.weight.data[0][0].numel()
        self.lr_mul = sqrt(2 / fan_in)

    def forward(self, x):
        x = x * self.lr_mul
        return self.conv(x)


class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, norm=False):

        super(ConvBlock, self).__init__()

        if norm:
            self.conv = nn.Sequential(
                EqualConv2d(in_ch, out_ch, kernel_size=3, padding=1),
                PixelNorm(),
                nn.LeakyReLU(0.1),
                EqualConv2d(out_ch, out_ch, kernel_size=3, padding=1),
                PixelNorm(),
                nn.LeakyReLU(0.1))
        else:
            self.conv = nn.Sequential(
                EqualConv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1))

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear')


class Downsample(nn.Module):
    def __init__(self, scale_factor=0.5, size=None):
        super(Downsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, size=self.size, mode='bilinear')


class ResnetFeatures(nn.Module):

    def __init__(self, bottleneck_filters):

        super(ResnetFeatures, self).__init__()

        resnet = resnet18(pretrained=True)

        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4)

        for layer in self.features[:-1]:  # freeze initial layers
            for param in layer.parameters():
                param.requires_grad = False

        self.register_buffer('means', torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        self.register_buffer('stds', torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))

        #self.conv = ConvBlock(512, bottleneck_filters, norm=True)

        self.conv = nn.Sequential(
            EqualConv2d(512, bottleneck_filters, kernel_size=3, padding=1),
            PixelNorm(),
            nn.LeakyReLU(0.1))

    def forward(self, x):

        x = (x + 1) / 2  # [0, 1]
        x = (x - self.means) / self.stds  # normalise as per imagenet inputs

        x = self.features(x)
        x = self.conv(x)

        return x


class Generator(nn.Module):

    def __init__(self, nb_filters, x_dim, z_dim, backbone=False):

        super(Generator, self).__init__()

        self.nb_filters = nb_filters
        self.bottleneck_res = 4
        self.bottleneck_filters = 256

        if backbone:

            self.encoder = ResnetFeatures(self.bottleneck_filters)

        else:  # no backbone - symmetric encoder

            self.encoder = nn.Sequential(
                ConvBlock(3, 2 * nb_filters, norm=True), Downsample(),
                ConvBlock(2 * nb_filters, 4 * nb_filters, norm=True), Downsample(),
                ConvBlock(4 * nb_filters, 8 * nb_filters, norm=True), Downsample(),
                ConvBlock(8 * nb_filters, 8 * nb_filters, norm=True), Downsample(),
                ConvBlock(8 * nb_filters, 4 * nb_filters, norm=True), Downsample(scale_factor=None, size=4))

        self.proj = nn.Linear(self.bottleneck_res ** 2 * self.bottleneck_filters, z_dim)
        self.deproj = nn.Sequential(nn.Linear(z_dim, nb_filters * 7 * 7),
                                    nn.LeakyReLU(0.1))

        self.decoder = nn.Sequential(
            Upsample(), ConvBlock(1 * nb_filters, 8 * nb_filters, norm=True),
            Upsample(), ConvBlock(8 * nb_filters, 8 * nb_filters, norm=True),
            Upsample(), ConvBlock(8 * nb_filters, 4 * nb_filters, norm=True),
            Upsample(), ConvBlock(4 * nb_filters, 2 * nb_filters, norm=True),
            Upsample(), ConvBlock(2 * nb_filters, 1 * nb_filters, norm=True))

        self.to_rgb = nn.ModuleList([
            EqualConv2d(1 * nb_filters, 3, 1),
            EqualConv2d(8 * nb_filters, 3, 1),
            EqualConv2d(8 * nb_filters, 3, 1),
            EqualConv2d(4 * nb_filters, 3, 1),
            EqualConv2d(2 * nb_filters, 3, 1),
            EqualConv2d(1 * nb_filters, 3, 1)])

    def fade_out(self, outputs, layer, rgb_1, rgb_2, alpha):

        skip_rgb = rgb_1(outputs)
        outputs = rgb_2(layer(outputs))

        return (1 - alpha) * skip_rgb + alpha * outputs

    def encode(self, x):

        x = self.encoder(x)
        x = x.view(x.shape[0], self.bottleneck_res ** 2 * self.bottleneck_filters)
        z = self.proj(x)

        return z

    def decode(self, z, step, alpha):

        x = self.deproj(z)
        x = x.view(x.shape[0], self.nb_filters, 7, 7)

        idx = 2 * int(step) - 1
        x = self.decoder[:idx](x)
        x = self.fade_out(x, self.decoder[idx], self.to_rgb[step-1], self.to_rgb[step], alpha)

        x = nn.Tanh()(x)

        return x

    def forward(self, x, step, alpha):

        z = self.encode(x)
        x = self.decode(z, step, alpha)

        return z, x


class Discriminator(nn.Module):

    def __init__(self, nb_filters=64):

        super(Discriminator, self).__init__()

        self.max_steps = 5

        self.from_rgb = nn.ModuleList([
            EqualConv2d(3, 8 * nb_filters, 1),
            EqualConv2d(3, 8 * nb_filters, 1),
            EqualConv2d(3, 8 * nb_filters, 1),
            EqualConv2d(3, 4 * nb_filters, 1),
            EqualConv2d(3, 2 * nb_filters, 1),
            EqualConv2d(3, 1 * nb_filters, 1)])

        self.encoder = nn.Sequential(
            ConvBlock(1 * nb_filters, 2 * nb_filters, norm=False), Downsample(),
            ConvBlock(2 * nb_filters, 4 * nb_filters, norm=False), Downsample(),
            ConvBlock(4 * nb_filters, 8 * nb_filters, norm=False), Downsample(),
            ConvBlock(8 * nb_filters, 8 * nb_filters, norm=False), Downsample(),
            ConvBlock(8 * nb_filters, 8 * nb_filters, norm=False), Downsample(),
            ConvBlock(8 * nb_filters, 8 * nb_filters, norm=False), Downsample(scale_factor=None, size=4),
            ConvBlock(8 * nb_filters, 1, norm=False))

    def fade_in(self, inputs, layer, rgb_1, rgb_2, alpha):

        skip_rgb = rgb_1(Downsample()(inputs))
        inputs = Downsample()(layer(rgb_2(inputs)))
        return (1 - alpha) * skip_rgb + alpha * inputs

    def forward(self, x, step, alpha):

        idx = 2 * (self.max_steps - step)

        x = self.fade_in(x, self.encoder[idx], self.from_rgb[step-1], self.from_rgb[step], alpha)
        x = self.encoder[idx+2:](x)

        return x


class Discriminator_Gauss(nn.Module):

    def __init__(self, z_dim):

        super(Discriminator_Gauss, self).__init__()

        self.validity = nn.Sequential(nn.Linear(z_dim, 512),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(0.2),
                                      nn.Linear(512, 512),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(0.2),
                                      nn.Linear(512, 1))

    def forward(self, x):
        return self.validity(x).squeeze()
