import torch
import torch.nn as nn

from .blocks import FeatureFusionBlock, _make_scratch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download


def copy_weights(from_model, to_model):
    # Get the state dictionary of the source model
    state_dict = from_model.state_dict()

    # Load the state dictionary into the target model
    to_model.load_state_dict(state_dict, strict=True)

def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

class DPTHead(nn.Module):
    def __init__(self, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        # out_channels = [in_channels // 8, in_channels // 4, in_channels // 2, in_channels]
        # out_channels = [in_channels // 4, in_channels // 2, in_channels, in_channels]
        # out_channels = [in_channels, in_channels, in_channels, in_channels]
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)) # Original x [ B, h//14 * w//14, C]
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
            
        return out


class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl', features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):

        super(DPT_DINOv2, self).__init__()

        torch.manual_seed(1)
        try:
            self.pretrained = torch.hub.load('../torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=False)
        except Exception as e:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))

        dim = self.pretrained.blocks[0].attn.qkv.in_features
        
        self.depth_head = DPTHead(dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

    def forward(self, x):
        h, w = x.shape[-2:]

        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        
        patch_h, patch_w = h // 14, w // 14

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth.squeeze(1)


class DPT_DINOv2_dual(nn.Module):
    def __init__(self, encoder='vitl', features=256, use_bn=False, out_channels=[256, 512, 1024, 1024],
                 use_clstoken=False, fusion_block=None):
        super(DPT_DINOv2_dual, self).__init__()

        torch.manual_seed(1)

        try:
            self.pretrained = torch.hub.load('./torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder),
                                             source='local', pretrained=False)
            self.pretrained_extra = torch.hub.load('./torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder),
                                             source='local', pretrained=False)
        except Exception as e:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))
            self.pretrained_extra = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))


        dim = self.pretrained.blocks[0].attn.qkv.in_features

        self.depth_head = DPTHead(dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        self.depth_head_extra = DPTHead(dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

        self.fusion_block = fusion_block

    def set_extras(self):
        copy_weights(self.pretrained, self.pretrained_extra)
        copy_weights(self.depth_head, self.depth_head_extra)

    def param_freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def fix_weights_extras(self):
        self.param_freeze(self.pretrained_extra)
        self.param_freeze(self.depth_head_extra)


    def forward(self, x, x_extra):
        h, w = x.shape[-2:]
        assert x.shape[-2:] == x_extra.shape[-2:], "Input shapes in DPT_DINOv2_dual must match"

        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        features_extra = self.pretrained_extra.get_intermediate_layers(x_extra, 4, return_class_token=True)

        # ### For debug, print the shape of the features
        # print("========================= The shape of features =========================")
        # for f in features: # Features is a list with length 4, the 4 elements are with the shape shape
        #     if type(f) == tuple:
        #         print("Shape of f[0] and f[1]: ", f[0].shape, f[1].shape) ## [B, H//14 x W//14, 1024] [B, 1024]
        #     else:
        #         print("Shape of f: ", f.shape)
        # print("=========================================================================")

        # Fuse features from both images
        if self.fusion_block is not None:
            features, features_extra = self.fusion_block(features, features_extra, return_class_token=True)

        patch_h, patch_w = h // 14, w // 14

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        depth_extra = self.depth_head_extra(features_extra, patch_h, patch_w)
        depth_extra = F.interpolate(depth_extra, size=(h, w), mode="bilinear", align_corners=True)
        depth_extra = F.relu(depth_extra)

        return depth.squeeze(1), depth_extra.squeeze(1), features[-1][0], features_extra[-1][0]