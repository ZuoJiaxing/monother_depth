# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import itertools
# from copy import deepcopy
import numpy as np

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from zoedepth.models.depth_model import DepthModel
from zoedepth.models.base_models.midas import MidasCore
from zoedepth.models.base_models.depth_anything import DepthAnythingCore, DepthAnythingCore_Dual
from zoedepth.models.layers.attractor import AttractorLayer, AttractorLayerUnnormed
from zoedepth.models.layers.dist_layers import ConditionalLogBinomial
from zoedepth.models.layers.localbins_layers import (Projector, SeedBinRegressor,
                                            SeedBinRegressorUnnormed)
from zoedepth.models.model_io import load_state_from_resource

from zoedepth.models.zoedepth_rgbt.fusion_block import attension_fusion_block, twopass_fusion_block, simple_twopass_fusion_block, simple_crossattn_fusion_block
from zoedepth.utils.depth_utils import util_add_row,colored_depthmap, cosine_similarity
import torchvision.transforms as VT
from zoedepth.utils.depth_utils import remap_pixel_coordinates, remap_depth_values
from torch.nn.functional import grid_sample
from zoedepth.trainers.loss import Nll_loss, get_mask_exlude_top20

def copy_weights(from_model, to_model):
    # Get the state dictionary of the source model
    state_dict = from_model.state_dict()

    # Load the state dictionary into the target model
    to_model.load_state_dict(state_dict, strict=True)


class ZoeDepth(DepthModel):
    def __init__(self, core,  n_bins=64, bin_centers_type="softplus", bin_embedding_dim=128, min_depth=1e-3, max_depth=10,
                 n_attractors=[16, 8, 4, 1], attractor_alpha=300, attractor_gamma=2, attractor_kind='sum', attractor_type='exp', min_temp=5, max_temp=50, train_midas=True,
                 midas_lr_factor=10, encoder_lr_factor=10, pos_enc_lr_factor=10, inverse_midas=False, **kwargs):
        """ZoeDepth model. This is the version of ZoeDepth that has a single metric head

        Args:
            core (models.base_models.midas.MidasCore): The base midas model that is used for extraction of "relative" features
            n_bins (int, optional): Number of bin centers. Defaults to 64.
            bin_centers_type (str, optional): "normed" or "softplus". Activation type used for bin centers. For "normed" bin centers, linear normalization trick is applied. This results in bounded bin centers.
                                               For "softplus", softplus activation is used and thus are unbounded. Defaults to "softplus".
            bin_embedding_dim (int, optional): bin embedding dimension. Defaults to 128.
            min_depth (float, optional): Lower bound for normed bin centers. Defaults to 1e-3.
            max_depth (float, optional): Upper bound for normed bin centers. Defaults to 10.
            n_attractors (List[int], optional): Number of bin attractors at decoder layers. Defaults to [16, 8, 4, 1].
            attractor_alpha (int, optional): Proportional attractor strength. Refer to models.layers.attractor for more details. Defaults to 300.
            attractor_gamma (int, optional): Exponential attractor strength. Refer to models.layers.attractor for more details. Defaults to 2.
            attractor_kind (str, optional): Attraction aggregation "sum" or "mean". Defaults to 'sum'.
            attractor_type (str, optional): Type of attractor to use; "inv" (Inverse attractor) or "exp" (Exponential attractor). Defaults to 'exp'.
            min_temp (int, optional): Lower bound for temperature of output probability distribution. Defaults to 5.
            max_temp (int, optional): Upper bound for temperature of output probability distribution. Defaults to 50.
            train_midas (bool, optional): Whether to train "core", the base midas model. Defaults to True.
            midas_lr_factor (int, optional): Learning rate reduction factor for base midas model except its encoder and positional encodings. Defaults to 10.
            encoder_lr_factor (int, optional): Learning rate reduction factor for the encoder in midas model. Defaults to 10.
            pos_enc_lr_factor (int, optional): Learning rate reduction factor for positional encodings in the base midas model. Defaults to 10.
        """
        super().__init__()

        self.core = core
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.min_temp = min_temp
        self.bin_centers_type = bin_centers_type

        self.midas_lr_factor = midas_lr_factor
        self.encoder_lr_factor = encoder_lr_factor
        self.pos_enc_lr_factor = pos_enc_lr_factor
        self.train_midas = train_midas
        self.inverse_midas = inverse_midas

        if self.encoder_lr_factor <= 0:
            self.core.freeze_encoder(
                freeze_rel_pos=self.pos_enc_lr_factor <= 0)

        N_MIDAS_OUT = 32
        btlnck_features = self.core.output_channels[0]
        num_out_features = self.core.output_channels[1:]

        # print('core output channels:', self.core.output_channels)
        
        self.conv2 = nn.Conv2d(btlnck_features, btlnck_features,
                               kernel_size=1, stride=1, padding=0)  # btlnck conv

        if bin_centers_type == "normed":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayer
        elif bin_centers_type == "softplus":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid1":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid2":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayer
        else:
            raise ValueError(
                "bin_centers_type should be one of 'normed', 'softplus', 'hybrid1', 'hybrid2'")

        self.seed_bin_regressor = SeedBinRegressorLayer(
            btlnck_features, n_bins=n_bins, min_depth=min_depth, max_depth=max_depth)
        self.seed_projector = Projector(btlnck_features, bin_embedding_dim)
        self.projectors = nn.ModuleList([
            Projector(num_out, bin_embedding_dim)
            for num_out in num_out_features
        ])
        self.attractors = nn.ModuleList([
            Attractor(bin_embedding_dim, n_bins, n_attractors=n_attractors[i], min_depth=min_depth, max_depth=max_depth,
                      alpha=attractor_alpha, gamma=attractor_gamma, kind=attractor_kind, attractor_type=attractor_type)
            for i in range(len(num_out_features))
        ])

        last_in = N_MIDAS_OUT + 1  # +1 for relative depth

        # use log binomial instead of softmax
        self.conditional_log_binomial = ConditionalLogBinomial(
            last_in, bin_embedding_dim, n_classes=n_bins, min_temp=min_temp, max_temp=max_temp)

    def forward(self, x, return_final_centers=False, denorm=False, return_probs=False, return_features=False, **kwargs):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)
            return_final_centers (bool, optional): Whether to return the final bin centers. Defaults to False.
            denorm (bool, optional): Whether to denormalize the input image. This reverses ImageNet normalization as midas normalization is different. Defaults to False.
            return_probs (bool, optional): Whether to return the output probability distribution. Defaults to False.
        
        Returns:
            dict: Dictionary containing the following keys:
                - rel_depth (torch.Tensor): Relative depth map of shape (B, H, W)
                - metric_depth (torch.Tensor): Metric depth map of shape (B, 1, H, W)
                - bin_centers (torch.Tensor): Bin centers of shape (B, n_bins). Present only if return_final_centers is True
                - probs (torch.Tensor): Output probability distribution of shape (B, n_bins, H, W). Present only if return_probs is True

        """
        # print('input shape', x.shape)
        
        b, c, h, w = x.shape
        # print("input shape:", x.shape)
        self.orig_input_width = w
        self.orig_input_height = h
        rel_depth, out = self.core(x, denorm=denorm, return_rel_depth=True)
        # print('-=-= rel_depth shape:', rel_depth.shape)
        # print('-=-= out type:', type(out))
        # print("-=-= out shapes:")
        # for k in range(len(out)):
        #     print(k, out[k].shape)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        ### Printed logs
        # -=-= len(output):  6
        # -=-= rel_depth shape: torch.Size([2, 392, 518])
        # -=-= out type: <class 'list'>
        # -=-= out shapes:
        # 0 torch.Size([2, 32, 392, 518])
        # 1 torch.Size([2, 256, 14, 19])
        # 2 torch.Size([2, 256, 28, 37])
        # 3 torch.Size([2, 256, 56, 74])
        # 4 torch.Size([2, 256, 112, 148])
        # 5 torch.Size([2, 256, 224, 296])

        
        outconv_activation = out[0]
        btlnck = out[1]
        x_blocks = out[2:]

        x_d0 = self.conv2(btlnck)
        x = x_d0
        _, seed_b_centers = self.seed_bin_regressor(x)

        if self.bin_centers_type == 'normed' or self.bin_centers_type == 'hybrid2':
            b_prev = (seed_b_centers - self.min_depth) / \
                (self.max_depth - self.min_depth)
        else:
            b_prev = seed_b_centers

        prev_b_embedding = self.seed_projector(x)

        # unroll this loop for better performance
        for projector, attractor, x in zip(self.projectors, self.attractors, x_blocks):
            b_embedding = projector(x)
            b, b_centers = attractor(
                b_embedding, b_prev, prev_b_embedding, interpolate=True)
            b_prev = b.clone()
            prev_b_embedding = b_embedding.clone()

        last = outconv_activation

        if self.inverse_midas:
            # invert depth followed by normalization
            rel_depth = 1.0 / (rel_depth + 1e-6)
            rel_depth = (rel_depth - rel_depth.min()) / \
                (rel_depth.max() - rel_depth.min())
        # concat rel depth with last. First interpolate rel depth to last size
        rel_cond = rel_depth.unsqueeze(1)
        rel_cond = nn.functional.interpolate(
            rel_cond, size=last.shape[2:], mode='bilinear', align_corners=True)
        last = torch.cat([last, rel_cond], dim=1)

        b_embedding = nn.functional.interpolate(
            b_embedding, last.shape[-2:], mode='bilinear', align_corners=True)
        x = self.conditional_log_binomial(last, b_embedding)

        # Now depth value is Sum px * cx , where cx are bin_centers from the last bin tensor
        # print(x.shape, b_centers.shape)
        b_centers = nn.functional.interpolate(
            b_centers, x.shape[-2:], mode='bilinear', align_corners=True)
        out = torch.sum(x * b_centers, dim=1, keepdim=True)

        # Structure output dict
        output = dict(metric_depth=out)
        if return_final_centers or return_probs:
            output['bin_centers'] = b_centers

        if return_probs:
            output['probs'] = x

        if return_features:
            output['features'] = torch.concat((last, b_embedding), dim=1) # (B, 33+ 128, H, W)

        return output

    def get_lr_params(self, lr):
        """
        Learning rate configuration for different layers of the model
        Args:
            lr (float) : Base learning rate
        Returns:
            list : list of parameters to optimize and their learning rates, in the format required by torch optimizers.
        """
        param_conf = []
        if self.train_midas:
            if self.encoder_lr_factor > 0:
                param_conf.append({'params': self.core.get_enc_params_except_rel_pos(
                ), 'lr': lr / self.encoder_lr_factor})

            if self.pos_enc_lr_factor > 0:
                param_conf.append(
                    {'params': self.core.get_rel_pos_params(), 'lr': lr / self.pos_enc_lr_factor})

            # midas_params = self.core.core.scratch.parameters()
            midas_params = self.core.core.depth_head.parameters()
            midas_lr_factor = self.midas_lr_factor
            param_conf.append(
                {'params': midas_params, 'lr': lr / midas_lr_factor})

        remaining_modules = []
        for name, child in self.named_children():
            if name != 'core':
                remaining_modules.append(child)
        remaining_params = itertools.chain(
            *[child.parameters() for child in remaining_modules])

        param_conf.append({'params': remaining_params, 'lr': lr})

        return param_conf

    @staticmethod
    def build(midas_model_type="DPT_BEiT_L_384", pretrained_resource=None, use_pretrained_midas=False, train_midas=False, freeze_midas_bn=True, **kwargs):
        # core = MidasCore.build(midas_model_type=midas_model_type, use_pretrained_midas=use_pretrained_midas,
        #                        train_midas=train_midas, fetch_features=True, freeze_bn=freeze_midas_bn, **kwargs)
        
        core = DepthAnythingCore.build(midas_model_type=midas_model_type, use_pretrained_midas=use_pretrained_midas,
                                       train_midas=train_midas, fetch_features=True, freeze_bn=freeze_midas_bn, **kwargs)
        
        model = ZoeDepth(core, **kwargs)
        if pretrained_resource:
            assert isinstance(pretrained_resource, str), "pretrained_resource must be a string"
            model = load_state_from_resource(model, pretrained_resource)
        return model

    @staticmethod
    def build_from_config(config):
        return ZoeDepth.build(**config)


class MatchingNet_stereorec(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_conv_layers=2, feat_window_size=5):
        super(MatchingNet_stereorec, self).__init__()

        layers = []
        for i in range(num_conv_layers):
            layers.append(
                nn.Conv1d(in_channels=input_dim if i == 0 else hidden_dim,
                          out_channels=hidden_dim,
                          kernel_size=3,
                          stride=2,  # Downsample by a factor of 2
                          padding=1)
            )
            layers.append(nn.ReLU())

        self.conv_layers = nn.Sequential(*layers)

        # Calculate the resulting length after the convolution and downsampling layers
        length_after_conv = feat_window_size
        for _ in range(num_conv_layers):
            length_after_conv = (length_after_conv + 2 * 1 - 3) // 2 + 1  # Formula for calculating output length

        half_hidden_dim = hidden_dim // 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * length_after_conv, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, half_hidden_dim),
            nn.ReLU(),
            nn.Linear(half_hidden_dim, 1)
        )

    def forward(self, x):
        "It takes input of shape (B, C, L) and returns output of shape (B, 1)"
        x = self.conv_layers(x) # (B, hidden_dim, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor # (B, hidden_dim*2)
        x = self.fc(x) # (B, 1)
        # x = torch.sigmoid(x)
        x =  1.0 / (1.0 + torch.exp(-x))
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

GRAD_CLIP = 0.1

class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        return grad_x.clamp(min=-0.01, max=0.01)

class GradientClip(nn.Module):
    def __init__(self):
        super(GradientClip, self).__init__()

    def forward(self, x):
        return GradClip.apply(x)

class Conf_UNet(nn.Module):
    def __init__(self, in_channels):
        super(Conf_UNet, self).__init__()
        self.in_channels = in_channels
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        # self.outc_unc = OutConv(64, 1)
        # self.outc_unc = nn.Sequential(
        #     nn.Linear(64, 1),
        #     GradientClip(),
        #     nn.Sigmoid())
        self.outc_unc = nn.Sequential(
            OutConv(64, 1),
            GradientClip(),
            nn.Sigmoid())

    def forward(self, x):
        """

        Args:
            x: B, C, H, W

        Returns:
            uncer: B, 1, H, W

        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        uncer = self.outc_unc(x)
        return uncer




class ZoeDepthDual(ZoeDepth):
    def __init__(self, core_dual,  n_bins=64, bin_centers_type="softplus", bin_embedding_dim=128, min_depth=1e-3, max_depth=10,
                 n_attractors=[16, 8, 4, 1], attractor_alpha=300, attractor_gamma=2, attractor_kind='sum', attractor_type='exp', min_temp=5, max_temp=50, train_midas=True,
                 midas_lr_factor=10, encoder_lr_factor=10, pos_enc_lr_factor=10, inverse_midas=False, **kwargs):
        """Compared to ZoeDepth model, this model has two branches for depth prediction. One is for RGB images, and the other one is for thermal images.

        Args:
            core (models.base_models.midas.MidasCore): The base midas model that is used for extraction of "relative" features
            n_bins (int, optional): Number of bin centers. Defaults to 64.
            bin_centers_type (str, optional): "normed" or "softplus". Activation type used for bin centers. For "normed" bin centers, linear normalization trick is applied. This results in bounded bin centers.
                                               For "softplus", softplus activation is used and thus are unbounded. Defaults to "softplus".
            bin_embedding_dim (int, optional): bin embedding dimension. Defaults to 128.
            min_depth (float, optional): Lower bound for normed bin centers. Defaults to 1e-3.
            max_depth (float, optional): Upper bound for normed bin centers. Defaults to 10.
            n_attractors (List[int], optional): Number of bin attractors at decoder layers. Defaults to [16, 8, 4, 1].
            attractor_alpha (int, optional): Proportional attractor strength. Refer to models.layers.attractor for more details. Defaults to 300.
            attractor_gamma (int, optional): Exponential attractor strength. Refer to models.layers.attractor for more details. Defaults to 2.
            attractor_kind (str, optional): Attraction aggregation "sum" or "mean". Defaults to 'sum'.
            attractor_type (str, optional): Type of attractor to use; "inv" (Inverse attractor) or "exp" (Exponential attractor). Defaults to 'exp'.
            min_temp (int, optional): Lower bound for temperature of output probability distribution. Defaults to 5.
            max_temp (int, optional): Upper bound for temperature of output probability distribution. Defaults to 50.
            train_midas (bool, optional): Whether to train "core", the base midas model. Defaults to True.
            midas_lr_factor (int, optional): Learning rate reduction factor for base midas model except its encoder and positional encodings. Defaults to 10.
            encoder_lr_factor (int, optional): Learning rate reduction factor for the encoder in midas model. Defaults to 10.
            pos_enc_lr_factor (int, optional): Learning rate reduction factor for positional encodings in the base midas model. Defaults to 10.
        """
        super().__init__(core_dual, n_bins=n_bins, bin_centers_type=bin_centers_type, bin_embedding_dim=bin_embedding_dim, min_depth=min_depth, max_depth=max_depth,
                         n_attractors=n_attractors, attractor_alpha=attractor_alpha, attractor_gamma=attractor_gamma, attractor_kind=attractor_kind, attractor_type=attractor_type, min_temp=min_temp, max_temp=max_temp, train_midas=train_midas,
                         midas_lr_factor=midas_lr_factor, encoder_lr_factor=encoder_lr_factor, pos_enc_lr_factor=pos_enc_lr_factor, inverse_midas=inverse_midas, **kwargs)

        # ### For extracting featup dino features
        # RGB_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        # RGB_DEFAULT_STD = (0.229, 0.224, 0.225)
        # THER_DEFAULT_MEAN = (0.45, 0.45, 0.45)
        # THER_DEFAULT_STD = (0.229, 0.224, 0.225)
        #
        # self.input_height = kwargs.get('input_height')
        # self.input_width = kwargs.get('input_width')
        # RESIZE_H = (self.input_height) // 16 * 16  # 16 for dino16, 14 for dinov2
        # RESIZE_W = (self.input_width) // 16 * 16
        # self.dino_ther_prepare_transform = VT.Compose([
        #     VT.Resize((RESIZE_H, RESIZE_W)),
        #     # T.ToTensor(),
        #     VT.Normalize(mean=RGB_DEFAULT_MEAN, std=RGB_DEFAULT_STD),
        # ])
        # self.dino_rgb_prepare_transform = VT.Compose([
        #     VT.Resize((RESIZE_H, RESIZE_W)),
        #     # T.ToTensor(),
        #     VT.Normalize(mean=THER_DEFAULT_MEAN, std=THER_DEFAULT_STD),
        # ])
        # self.feat_upsampler = torch.hub.load("mhamilton723/FeatUp", 'dino16', use_norm=False) # dino16, dinov2 # Refer to https://github.com/mhamilton723/FeatUp
        # # Free the weights of self.feat_upsampler
        # for param in self.feat_upsampler.parameters():
        #     param.requires_grad = False


        self.core = core_dual
        btlnck_features = self.core.output_channels[0]
        num_out_features = self.core.output_channels[1:]
        self.conv2_extra = nn.Conv2d(btlnck_features, btlnck_features,
                                     kernel_size=1, stride=1, padding=0)  # btlnck conv

        if bin_centers_type == "normed":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayer
        elif bin_centers_type == "softplus":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid1":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid2":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayer
        else:
            raise ValueError(
                "bin_centers_type should be one of 'normed', 'softplus', 'hybrid1', 'hybrid2'")

        self.seed_bin_regressor_extra = SeedBinRegressorLayer(
            btlnck_features, n_bins=n_bins, min_depth=min_depth, max_depth=max_depth)

        self.seed_projector_extra = Projector(btlnck_features, bin_embedding_dim)

        self.attractors_extra = nn.ModuleList([
            Attractor(bin_embedding_dim, n_bins, n_attractors=n_attractors[i], min_depth=min_depth, max_depth=max_depth,
                      alpha=attractor_alpha, gamma=attractor_gamma, kind=attractor_kind, attractor_type=attractor_type)
            for i in range(len(num_out_features))
        ])

        self.projectors_extra = nn.ModuleList([
            Projector(num_out, bin_embedding_dim)
            for num_out in num_out_features
        ])

        N_MIDAS_OUT = 32
        last_in = N_MIDAS_OUT + 1  # +1 for relative depth
        # use log binomial instead of softmax
        self.conditional_log_binomial_extra = ConditionalLogBinomial(last_in, bin_embedding_dim, n_classes=n_bins, min_temp=min_temp, max_temp=max_temp)

        self.teachstudent_uncer = kwargs.get('teachstudent_uncer', False)
        if self.teachstudent_uncer:
            self.uncer_feat_input_dim = 161*2 # (33 + 128)*2
            self.conf_net = Conf_UNet(in_channels=8)
            self.nll_loss = Nll_loss()

    def set_extras(self):
        self.core.set_extras()
        copy_weights(self.conv2, self.conv2_extra)
        copy_weights(self.seed_bin_regressor, self.seed_bin_regressor_extra)
        copy_weights(self.seed_projector, self.seed_projector_extra)
        copy_weights(self.conditional_log_binomial, self.conditional_log_binomial_extra)
        for i in range(len(self.projectors)):
            copy_weights(self.projectors[i], self.projectors_extra[i])
            copy_weights(self.attractors[i], self.attractors_extra[i])

    def param_freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def fix_weights_extras(self):
        self.core.fix_weights_extras()
        self.param_freeze(self.conv2_extra)
        self.param_freeze(self.seed_bin_regressor_extra)
        self.param_freeze(self.seed_projector_extra)
        self.param_freeze(self.conditional_log_binomial_extra)
        for i in range(len(self.projectors_extra)):
            self.param_freeze(self.projectors_extra[i])
            self.param_freeze(self.attractors_extra[i])
        if self.teachstudent_uncer:
            self.param_freeze(self.conf_net)

    def set_attach_hooks(self):
        self.core.set_attach_hooks()

    def forward_last_part_x(self, rel_depth, out, return_final_centers, return_probs, return_features=False):
        outconv_activation = out[0]
        btlnck = out[1]
        x_blocks = out[2:-1] # out[2:-1]
        x_feat=out[-1].clone() # For debug only

        x_d0 = self.conv2(btlnck)
        x = x_d0
        _, seed_b_centers = self.seed_bin_regressor(x)

        if self.bin_centers_type == 'normed' or self.bin_centers_type == 'hybrid2':
            b_prev = (seed_b_centers - self.min_depth) / \
                     (self.max_depth - self.min_depth)
        else:
            b_prev = seed_b_centers

        prev_b_embedding = self.seed_projector(x)

        # unroll this loop for better performance
        for projector, attractor, x in zip(self.projectors, self.attractors, x_blocks):
            b_embedding = projector(x)
            b, b_centers = attractor(
                b_embedding, b_prev, prev_b_embedding, interpolate=True)
            b_prev = b.clone()
            prev_b_embedding = b_embedding.clone()

        last = outconv_activation

        if self.inverse_midas:
            # invert depth followed by normalization
            rel_depth = 1.0 / (rel_depth + 1e-6)
            rel_depth = (rel_depth - rel_depth.min()) / \
                        (rel_depth.max() - rel_depth.min())
        # concat rel depth with last. First interpolate rel depth to last size
        rel_cond = rel_depth.unsqueeze(1)
        rel_cond = nn.functional.interpolate(
            rel_cond, size=last.shape[2:], mode='bilinear', align_corners=True)
        last = torch.cat([last, rel_cond], dim=1)

        b_embedding = nn.functional.interpolate(
            b_embedding, last.shape[-2:], mode='bilinear', align_corners=True)
        x = self.conditional_log_binomial(last, b_embedding)

        # Now depth value is Sum px * cx , where cx are bin_centers from the last bin tensor
        # print(x.shape, b_centers.shape)
        b_centers = nn.functional.interpolate(
            b_centers, x.shape[-2:], mode='bilinear', align_corners=True)
        out = torch.sum(x * b_centers, dim=1, keepdim=True)

        # Structure output dict
        output = dict(metric_depth=out)
        if return_final_centers or return_probs:
            output['bin_centers'] = b_centers

        if return_probs:
            output['probs'] = x

        if return_features:
            # output['features'] = x_feat
            output['features'] = torch.concat((last, b_embedding), dim=1) # (B, 33+ 128, H, W)

        return output

    def forward_last_part_y(self, rel_depth, out, return_final_centers, return_probs, return_features=False):
        outconv_activation = out[0]
        btlnck = out[1]
        x_blocks = out[2:-1] # out[2:-1]
        x_feat=out[-1].clone() # For debug only

        x_d0 = self.conv2_extra(btlnck)
        x = x_d0
        _, seed_b_centers = self.seed_bin_regressor_extra(x)

        if self.bin_centers_type == 'normed' or self.bin_centers_type == 'hybrid2':
            b_prev = (seed_b_centers - self.min_depth) / \
                     (self.max_depth - self.min_depth)
        else:
            b_prev = seed_b_centers

        prev_b_embedding = self.seed_projector_extra(x)

        # unroll this loop for better performance
        for projector, attractor, x in zip(self.projectors_extra, self.attractors_extra, x_blocks):
            b_embedding = projector(x)
            b, b_centers = attractor(
                b_embedding, b_prev, prev_b_embedding, interpolate=True)
            b_prev = b.clone()
            prev_b_embedding = b_embedding.clone()

        last = outconv_activation

        if self.inverse_midas:
            # invert depth followed by normalization
            rel_depth = 1.0 / (rel_depth + 1e-6)
            rel_depth = (rel_depth - rel_depth.min()) / \
                        (rel_depth.max() - rel_depth.min())
        # concat rel depth with last. First interpolate rel depth to last size
        rel_cond = rel_depth.unsqueeze(1)
        rel_cond = nn.functional.interpolate(
            rel_cond, size=last.shape[2:], mode='bilinear', align_corners=True)
        last = torch.cat([last, rel_cond], dim=1)

        b_embedding = nn.functional.interpolate(
            b_embedding, last.shape[-2:], mode='bilinear', align_corners=True)
        x = self.conditional_log_binomial_extra(last, b_embedding)

        # print("$$$$$$$$$$ For debug, x.shape, b_centers.shape: ", x.shape, b_centers.shape) # [1, 64, 196, 252], [1, 64, 112, 144]

        # Now depth value is Sum px * cx , where cx are bin_centers from the last bin tensor
        # print(x.shape, b_centers.shape)
        b_centers = nn.functional.interpolate(
            b_centers, x.shape[-2:], mode='bilinear', align_corners=True)
        out = torch.sum(x * b_centers, dim=1, keepdim=True)

        # Structure output dict
        output = dict(metric_depth=out)
        if return_final_centers or return_probs:
            output['bin_centers'] = b_centers

        if return_probs:
            output['probs'] = x

        if return_features:
            # output['features'] = x_feat
            output['features'] = torch.concat((last, b_embedding), dim=1) # (B, 33+ 128, H, W)

        return output

    def warpa_sampleb_depth(self, pred_depths_a, pred_depths_b, intrinsics_a, intrinsics_b, pose_cam_a2b=None):
        pred_w, pred_h = pred_depths_b.shape[-1], pred_depths_b.shape[-2]
        pred_depths_a_clone = pred_depths_a.permute(0, 2, 3, 1).contiguous().clone()  # (B, C, H, W) -> (B, H, W, C)
        apix_in_b, adepth_in_b = remap_pixel_coordinates(pred_depths_a_clone, intrinsics_a, intrinsics_b, pose_cam1tocam2=pose_cam_a2b)  # pose_t2rgb
        apix_in_b = apix_in_b.detach()
        adepth_in_b = adepth_in_b.permute(0, 3, 1, 2).contiguous()  # (B, H, W, 1) -> (B, 1, H, W)
        apix_in_b = torch.stack([2 * apix_in_b[:, :, :, 0] / (pred_w - 1) - 1, 2 * apix_in_b[:, :, :, 1] / (pred_h - 1) - 1], dim=-1)
        b_depth_c2a = grid_sample(pred_depths_b, apix_in_b, mode='bilinear', padding_mode='zeros', align_corners=True)
        return apix_in_b, adepth_in_b, b_depth_c2a

    # bpix_in_a, bdepth_in_a, a_depth_c2b # All of them are in cam_a coordinate
    def get_sampleda_depth_inb(self, bpix_in_a, a_depth_c2b, intrinsics_a, intrinsics_b, pose_cam_a2b=None):
        """
        bpix_in_a: (B, H, W, 2) is the normalized pixel coordinates [-1, 1]
        """
        pred_w, pred_h = a_depth_c2b.shape[-1], a_depth_c2b.shape[-2]
        a_depth_c2b_clone = a_depth_c2b.permute(0, 2, 3, 1).contiguous().clone()  # (B, C, H, W) -> (B, H, W, C)
        u_bpix_in_a = (bpix_in_a[:, :, :, 0] + 1) * (pred_w - 1) / 2
        v_bpix_in_a = (bpix_in_a[:, :, :, 1] + 1) * (pred_h - 1) / 2
        uv_bpix_in_a = torch.stack([u_bpix_in_a, v_bpix_in_a], dim=-1) # Note that this might out of image range
        pix_in_b, depth_in_b = remap_depth_values(a_depth_c2b_clone, uv_bpix_in_a, intrinsics_a, intrinsics_b, pose_cam1tocam2=pose_cam_a2b)  # pose_t2rgb
        # pix_in_b = pix_in_b.detach() # (B, H, W, 2)
        depth_in_b = depth_in_b.permute(0, 3, 1, 2).contiguous()  # (B, H, W, 1) -> (B, 1, H, W)
        return depth_in_b

    def forward(self, x, y, return_final_centers=False, denorm=False, return_probs=False, nll_loss=False,
                intrinsics_x=None, intrinsics_y=None, extrinsincs_x2y=None, **kwargs):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W); thermal image
            y (torch.Tensor): Input extra image tensor of shape (B, C, H, W);
            return_final_centers (bool, optional): Whether to return the final bin centers. Defaults to False.
            denorm (bool, optional): Whether to denormalize the input image. This reverses ImageNet normalization as midas normalization is different. Defaults to False.
            return_probs (bool, optional): Whether to return the output probability distribution. Defaults to False.

        Returns:
            dict: Dictionary containing the following keys:
                - rel_depth (torch.Tensor): Relative depth map of shape (B, H, W)
                - metric_depth (torch.Tensor): Metric depth map of shape (B, 1, H, W)
                - bin_centers (torch.Tensor): Bin centers of shape (B, n_bins). Present only if return_final_centers is True
                - probs (torch.Tensor): Output probability distribution of shape (B, n_bins, H, W). Present only if return_probs is True

        """
        # print('input shape', x.shape)

        b, c, h, w = x.shape
        assert x.shape == y.shape, "Input images for ZoeDepthDual must have the same shape"
        # print("input shape:", x.shape)
        self.orig_input_width = w
        self.orig_input_height = h
        # print("~~~~~~~~~~~~~~~~~ For debug, ZoeDepthDual, input,  x.shape, y.shape: ", x.shape, y.shape)
        rel_depth, rel_depth_extra, out, out_extra = self.core(x, y, denorm=denorm, return_rel_depth=True)
        # print('-=-= rel_depth shape:', rel_depth.shape)
        # print('-=-= out type:', type(out))
        # print("-=-= out shapes:")
        # for k in range(len(out)):
        #     print(k, out[k].shape)
        # print("~~~~~~~~~~~~~~~~~ For debug, ZoeDepthDual, out,  x.shape, y.shape: ", x.shape, y.shape)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        ### Printed logs
        # -=-= len(output):  6
        # -=-= rel_depth shape: torch.Size([2, 392, 518])
        # -=-= out type: <class 'list'>
        # -=-= out shapes:
        # 0 torch.Size([2, 32, 392, 518])
        # 1 torch.Size([2, 256, 14, 19])
        # 2 torch.Size([2, 256, 28, 37])
        # 3 torch.Size([2, 256, 56, 74])
        # 4 torch.Size([2, 256, 112, 148])
        # 5 torch.Size([2, 256, 224, 296])

        # # ### Debug, visualize the feature map
        # from featup.plotting import plot_feats   #, plot_feats_two
        # # plot_feats_two(x[0], out[0][0], out_extra[0][0], y[0], out[5][0], out_extra[5][0])
        #
        output_x = self.forward_last_part_x(rel_depth, out, return_final_centers, return_probs, return_features = nll_loss)
        output_y = self.forward_last_part_y(rel_depth_extra, out_extra, return_final_centers, return_probs, return_features = nll_loss)

        output_x['rel_depth'] = rel_depth
        output_y['rel_depth'] = rel_depth_extra



        # print("============= For debug, x[0].shape, output_x['features'][0].shape, output_y['features'][0].shape: ", x[0].shape, output_x['features'][0].shape, output_y['features'][0].shape)
        # plot_feats(x[0], output_x['features'][0], output_y['features'][0]) # (B, 1024, H//14, W//14)
        #
        # # assert (self.input_width == w) & (self.input_height == h), f"{self.input_width} != {w} or {self.input_height} != {h}"
        # ther_hr_feats = self.feat_upsampler(self.dino_ther_prepare_transform(x))  # (1, 384, H, W)
        # rgb_hr_feats = self.feat_upsampler(self.dino_rgb_prepare_transform(y))  # (1, 384, H, W)
        # plot_feats(x[0], ther_hr_feats[0], rgb_hr_feats[0])
        # output_x['features'] = ther_hr_feats
        # output_y['features'] = rgb_hr_feats

        if nll_loss:
            assert self.teachstudent_uncer
            assert intrinsics_x is not None
            assert intrinsics_y is not None
            if intrinsics_x.shape[-1] == 3: # It is a 3x3 matrix, instead of 3x4 projection matrix
                assert extrinsincs_x2y is not None
            extrinsincs_y2x = None
            if extrinsincs_x2y is not None:
                extrinsincs_y2x = torch.inverse(extrinsincs_x2y)

            pred_depths_x = output_x['metric_depth'] # (B, 1, H, W)
            pred_depths_y = output_y['metric_depth'] # (B, 1, H, W)
            pred_w, pred_h = pred_depths_x.shape[-1], pred_depths_x.shape[-2]
            if pred_depths_x.shape[-2:] != x.shape[-2:]:
                scale_h, scale_w = pred_h / h, pred_w / w
                intrinsics_x[:, 0, :] = intrinsics_x[:, 0, :] * scale_w
                intrinsics_x[:, 1, :] = intrinsics_x[:, 1, :] * scale_h
                intrinsics_y[:, 0, :] = intrinsics_y[:, 0, :] * scale_w
                intrinsics_y[:, 1, :] = intrinsics_y[:, 1, :] * scale_h
                y_resized = nn.functional.interpolate(y, pred_depths_y.shape[-2:], mode='bilinear', align_corners=True)

            # apix_in_b (B, H, W, 2), adepth_in_b (B, 1, H, W), b_depth_c2a (B, 1, H, W)
            apix_in_b, adepth_in_b, b_depth_c2a = self.warpa_sampleb_depth(pred_depths_x, pred_depths_y, intrinsics_x, intrinsics_y, pose_cam_a2b=extrinsincs_x2y)
            bpix_in_a, bdepth_in_a, a_depth_c2b = self.warpa_sampleb_depth(pred_depths_y, pred_depths_x, intrinsics_y, intrinsics_x, pose_cam_a2b=extrinsincs_y2x)
            # invalid_mask_bdepth_in_a = (bpix_in_a[..., 0] < -1) | (bpix_in_a[..., 0] > 1) | (bpix_in_a[..., 1] < -1) | (bpix_in_a[..., 1] > 1)
            # invalid_mask_bdepth_in_a = invalid_mask_bdepth_in_a.unsqueeze(1)
            # bdepth_in_a[invalid_mask_bdepth_in_a] = 0.0

            geta_depth_in_b = self.get_sampleda_depth_inb(bpix_in_a, a_depth_c2b, intrinsics_x, intrinsics_y, pose_cam_a2b=extrinsincs_x2y)
            diff_depth = (pred_depths_y.detach() - geta_depth_in_b).abs()

            feat_a = output_x['features'].detach() # (B, 161, H, W)
            feat_b = output_y['features'].detach() # (B, 161, H, W)
            feat_ch = feat_a.shape[1]

            b_feat_c2a = grid_sample(feat_b, apix_in_b.detach(), mode='bilinear', padding_mode='zeros', align_corners=True) # (B, C, H, W)
            a_feat_c2b = grid_sample(feat_a, bpix_in_a.detach(), mode='bilinear', padding_mode='zeros', align_corners=True) # (B, C, H, W)

            S_b = cosine_similarity(feat_b, a_feat_c2b) # (B, 1, H, W)
            S_a = cosine_similarity(feat_a, b_feat_c2a) # (B, 1, H, W)
            S_a_c2b = grid_sample(S_a, bpix_in_a.detach(), mode='nearest', padding_mode='zeros', align_corners=True)
            # diff_depth = (bdepth_in_a.detach() - a_depth_c2b).abs()
            exclude_top_diff_mask = get_mask_exlude_top20(diff_depth, 0.2)
            mask_diff_depth = (pred_depths_y > self.min_depth) & (geta_depth_in_b > self.min_depth)
            diff_depth[~mask_diff_depth] = -1.0
            # print("============= For debug, mean, std, meadian, min and max of diff_depth: ", diff_depth[mask_diff_depth].mean().item(), diff_depth[mask_diff_depth].std().item(), diff_depth[mask_diff_depth].median().item(), diff_depth[mask_diff_depth].min().item(), diff_depth[mask_diff_depth].max().item())

            # a_img_c2b = grid_sample(x, bpix_in_a.detach(), mode='bilinear', padding_mode='zeros', align_corners=True)


            mask_diff_depth = mask_diff_depth & (pred_depths_y < self.max_depth) & (geta_depth_in_b < self.max_depth)
            geta_depth_in_b[geta_depth_in_b < self.min_depth]  = 0.0
            input_uncernet = torch.cat([S_b, S_a_c2b, diff_depth, geta_depth_in_b, pred_depths_y, y_resized], dim=1) # y might be with different shape from pred_depths_y
            pred_depths_y_conf = self.conf_net(input_uncernet.detach()) # (B, 1, H, W)
            # print("------------- For debug, mean, std, meadian, min and max of pred_depths_y_conf: ", pred_depths_y_conf[mask_diff_depth].mean().item(), pred_depths_y_conf[mask_diff_depth].std().item(), pred_depths_y_conf[mask_diff_depth].median().item(), pred_depths_y_conf[mask_diff_depth].min().item(), pred_depths_y_conf[mask_diff_depth].max().item())
            output_y['conf'] = pred_depths_y_conf
            S_b_mask = ~ get_mask_exlude_top20(S_b, 0.8) # Keep the top 80%
            S_b_mask = (S_b > 1e-2) & S_b_mask
            consistency_mask = (diff_depth > 1e-2) & S_b_mask  & (pred_depths_y_conf > 1e-3) # & (S_b > S_a_c2b)  # TODO: tune this

            # deno = consistency_mask.numel()
            ### Debug: 0.99, 0.8, 0.999, 0.56, 0.7 ~0.8
            # print("------------- debug, raio of threshold on nll_l1 mask: ", (diff_depth > 1e-2).sum().item()/deno, (S_b_mask).sum().item()/deno, (pred_depths_y_conf > 1e-3).sum().item()/deno, (S_b > S_a_c2b).sum().item()/deno,  consistency_mask.sum().item()/deno)

            consistency_mask = (mask_diff_depth & consistency_mask & exclude_top_diff_mask).detach()
            consistency_residual = pred_depths_y_conf.detach() * diff_depth
            output_y['consistency_mask'] = consistency_mask
            output_y['consistency_l1loss'] =  consistency_residual[consistency_mask].mean()

            output_x['conf'] = None
            output_x['consistency_mask'] = None
            output_x['consistency_l1loss'] = 0.0







        # ########################## For debug, display the predicted relative depth and the predicted depth
        # print("rel_depth[0] shape, mean and std: ", rel_depth[0].shape, rel_depth[0].mean().item(), rel_depth[0].std().item(), rel_depth[0].min().item(), rel_depth[0].max().item())
        # print("rel_depth_extra[0] shape, mean and std: ", rel_depth_extra[0].shape, rel_depth_extra[0].mean().item(), rel_depth_extra[0].std().item(),
        #       rel_depth_extra[0].min().item(), rel_depth_extra[0].max().item())
        #
        # #### For debug, display
        # vis_img = 255.0*x[0].detach().cpu().numpy()
        # vis_img = vis_img.transpose(1, 2, 0).astype(np.uint8)
        # vis_rel_depth = colored_depthmap(rel_depth[0].detach().squeeze().cpu().numpy(), 0, 60).astype(np.uint8)
        # vis_img = cv2.resize(vis_img, (vis_rel_depth.shape[1], vis_rel_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        #
        # vis_img_rgb = 255.0*y[0].detach().cpu().numpy()
        # vis_img_rgb = vis_img_rgb.transpose(1, 2, 0).astype(np.uint8)
        # vis_rel_depth_rgb = colored_depthmap(rel_depth_extra[0].detach().squeeze().cpu().numpy(), 0, 60).astype(np.uint8)
        # vis_img_rgb = cv2.resize(vis_img_rgb, (vis_rel_depth.shape[1], vis_rel_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        #
        # # print("$$$$$$$$$$$$$$$ shape of vis_img, vis_rel_depth, vis_img_rgb, vis_rel_depth_rgb: ", vis_img.shape, vis_rel_depth.shape, vis_img_rgb.shape, vis_rel_depth_rgb.shape)
        # vis_total = np.hstack((vis_img, vis_rel_depth, vis_img_rgb, vis_rel_depth_rgb))
        # cv2.imshow("vis_total", vis_total)
        # cv2.waitKey()

        return output_x, output_y

    def get_lr_params(self, lr):
        """
        Learning rate configuration for different layers of the model
        Args:
            lr (float) : Base learning rate
        Returns:
            list : list of parameters to optimize and their learning rates, in the format required by torch optimizers.
        """
        param_conf = []
        if self.train_midas:
            if self.encoder_lr_factor > 0:
                param_conf.append({'params': self.core.get_enc_params_except_rel_pos(
                ), 'lr': lr / self.encoder_lr_factor})

            if self.pos_enc_lr_factor > 0:
                param_conf.append(
                    {'params': self.core.get_rel_pos_params(), 'lr': lr / self.pos_enc_lr_factor})

            # midas_params = self.core.core.scratch.parameters()
            midas_params = self.core.core.depth_head.parameters()
            midas_lr_factor = self.midas_lr_factor
            param_conf.append(
                {'params': midas_params, 'lr': lr / midas_lr_factor})

            midas_extra_params = self.core.core.depth_head_extra.parameters()
            param_conf.append(
                {'params': midas_extra_params, 'lr': lr / midas_lr_factor})

        remaining_modules = []
        for name, child in self.named_children():
            if name != 'core':
                remaining_modules.append(child)
        remaining_params = itertools.chain(
            *[child.parameters() for child in remaining_modules])

        param_conf.append({'params': remaining_params, 'lr': lr})

        return param_conf

    @staticmethod
    def build(midas_model_type="DPT_BEiT_L_384", pretrained_resource=None, use_pretrained_midas=False,
              train_midas=False, freeze_midas_bn=True, fusion_block_name=None, **kwargs):

        fusion_block = None
        if fusion_block_name == "linear_attention":
            print("[fusion_block]: linear_attention")
            fusion_block = attension_fusion_block([1024, 1024, 1024, 1024], n_layer=2)
        elif fusion_block_name == "simple_cross_attention":
            print("[fusion_block]: simple_cross_attention")
            fusion_block = simple_crossattn_fusion_block([1024, 1024, 1024, 1024], n_layer=2)
        elif fusion_block_name == "two_pass":
            print("[fusion_block]: two_pass")
            fusion_block = twopass_fusion_block([1024, 1024, 1024, 1024], n_layer=1)
        elif fusion_block_name == "simple_two_pass":
            print("[fusion_block]: simple_two_pass")
            fusion_block = simple_twopass_fusion_block([1024, 1024, 1024, 1024], n_layer=1)
        else:
            print("[fusion_block]: None")

        core_dual = DepthAnythingCore_Dual.build(midas_model_type=midas_model_type, use_pretrained_midas=use_pretrained_midas,
                                       train_midas=train_midas, fetch_features=True, freeze_bn=freeze_midas_bn, fusion_block=fusion_block,
                                       **kwargs)
        model = ZoeDepthDual(core_dual, **kwargs)
        if pretrained_resource:
            assert isinstance(pretrained_resource, str), "pretrained_resource must be a string"
            model = load_state_from_resource(model, pretrained_resource)

            if "ZoeDepthDual" not in pretrained_resource:
                print("---------------- ZoeDepthDual is not in pretrained_resource, calling set_extras() ----------------")
                model.set_extras()
            else:
                print("---------------- ZoeDepthDual is in pretrained_resource, using the loaded weights ----------------")

            # #### For debug only:
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Model, Print when load ckpt $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            # m = model
            # print("train_midas: ", train_midas)
            # print("Debug only, print1: \n", m.state_dict()['core.core.pretrained.blocks.0.mlp.fc2.weight'])
            # print("Debug only, print2: \n", m.state_dict()['core.core.depth_head.scratch.output_conv2.2.weight'])
            # print("Debug only, print3: \n", m.state_dict()['core.core.pretrained_extra.blocks.0.mlp.fc2.weight'])
            # print("Debug only, print4: \n", m.state_dict()['core.core.depth_head_extra.scratch.output_conv2.2.weight'])
            # print("Debug only, print5: \n", m.state_dict()['projectors.0._net.0.weight'])
            # print("Debug only, print6: \n", m.state_dict()['projectors_extra.0._net.0.weight'])
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        ### Fix the weights of the extra branch (RGB)
        freeze_weights_extra_branch = kwargs.get('freeze_weights_extra_branch', False)
        if freeze_weights_extra_branch:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX freeze_weights_extra_branch XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            model.fix_weights_extras()

        return model

    @staticmethod
    def build_from_config(config):
        return ZoeDepthDual.build(**config)