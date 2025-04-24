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

import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import Normalize
from zoedepth.models.base_models.dpt_dinov2.dpt import DPT_DINOv2
from zoedepth.models.base_models.dpt_dinov2.dpt import DPT_DINOv2_dual


def denormalize(x):
    """Reverses the imagenet normalization applied to the input.

    Args:
        x (torch.Tensor - shape(N,3,H,W)): input tensor

    Returns:
        torch.Tensor - shape(N,3,H,W): Denormalized input
    """
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return x * std + mean

def get_activation(name, bank):
    def hook(model, input, output):
        bank[name] = output
    return hook


class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
    ):
        """Init.
        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        print("Params passed to Resize transform:")
        print("\twidth: ", width)
        print("\theight: ", height)
        print("\tresize_target: ", resize_target)
        print("\tkeep_aspect_ratio: ", keep_aspect_ratio)
        print("\tensure_multiple_of: ", ensure_multiple_of)
        print("\tresize_method: ", resize_method)

        self.__width = width
        self.__height = height

        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of)
                 * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of)
                 * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(
                f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, x):
        width, height = self.get_size(*x.shape[-2:][::-1])
        return nn.functional.interpolate(x, (height, width), mode='bilinear', align_corners=True)

class PrepForMidas(object):
    def __init__(self, resize_mode="minimal", keep_aspect_ratio=True, img_size=384, do_resize=True, b_thermal=False):
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        net_h, net_w = img_size
        if b_thermal:
            self.normalization = Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        else:
            # self.normalization = Normalize(
            #     mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            self.normalization = Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resizer = Resize(net_w, net_h, keep_aspect_ratio=keep_aspect_ratio, ensure_multiple_of=14, resize_method=resize_mode) \
            if do_resize else nn.Identity()

    def __call__(self, x):
        return self.normalization(self.resizer(x))


class PrepForMidas_Dual(object):
    def __init__(self, resize_mode="minimal", keep_aspect_ratio=True, img_size=384, do_resize=True):
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        net_h, net_w = img_size
        self.normalization_thermal = Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        self.normalization_rgb = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resizer = Resize(net_w, net_h, keep_aspect_ratio=keep_aspect_ratio, ensure_multiple_of=14, resize_method=resize_mode) \
            if do_resize else nn.Identity()

    def __call__(self, thermal, rgb):
        return self.normalization_thermal(self.resizer(thermal)), self.normalization_rgb(self.resizer(rgb))


class DepthAnythingCore(nn.Module):
    def __init__(self, midas, trainable=False, fetch_features=True, layer_names=('out_conv', 'l4_rn', 'r4', 'r3', 'r2', 'r1'), freeze_bn=False, keep_aspect_ratio=True,
                 img_size=384, **kwargs):
        """Midas Base model used for multi-scale feature extraction.

        Args:
            midas (torch.nn.Module): Midas model.
            trainable (bool, optional): Train midas model. Defaults to False.
            fetch_features (bool, optional): Extract multi-scale features. Defaults to True.
            layer_names (tuple, optional): Layers used for feature extraction. Order = (head output features, last layer features, ...decoder features). Defaults to ('out_conv', 'l4_rn', 'r4', 'r3', 'r2', 'r1').
            freeze_bn (bool, optional): Freeze BatchNorm. Generally results in better finetuning performance. Defaults to False.
            keep_aspect_ratio (bool, optional): Keep the aspect ratio of input images while resizing. Defaults to True.
            img_size (int, tuple, optional): Input resolution. Defaults to 384.
        """
        super().__init__()
        self.core = midas
        self.output_channels = None
        self.core_out = {}
        self.trainable = trainable
        self.fetch_features = fetch_features
        # midas.scratch.output_conv = nn.Identity()
        self.handles = []
        # self.layer_names = ['out_conv','l4_rn', 'r4', 'r3', 'r2', 'r1']
        self.layer_names = layer_names

        self.set_trainable(trainable)
        self.set_fetch_features(fetch_features)
        b_thermal = kwargs.get('dataloader_type', 0) > 0
        self.prep = PrepForMidas(keep_aspect_ratio=keep_aspect_ratio,
                                 img_size=img_size, do_resize=kwargs.get('do_resize', True), b_thermal=b_thermal)

        if freeze_bn:
            self.freeze_bn()

    def set_trainable(self, trainable):
        self.trainable = trainable
        if trainable:
            self.unfreeze()
        else:
            self.freeze()
        return self

    def set_fetch_features(self, fetch_features):
        self.fetch_features = fetch_features
        if fetch_features:
            if len(self.handles) == 0:
                self.attach_hooks(self.core)
        else:
            self.remove_hooks()
        return self

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.trainable = False
        return self

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self.trainable = True
        return self

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x, denorm=False, return_rel_depth=False):
        # print('input to midas:', x.shape)
        with torch.no_grad():
            if denorm:
                x = denormalize(x)
            x = self.prep(x)
        
        with torch.set_grad_enabled(self.trainable):

            rel_depth = self.core(x)
            if not self.fetch_features:
                return rel_depth
        out = [self.core_out[k] for k in self.layer_names]

        if return_rel_depth:
            return rel_depth, out
        return out

    def get_rel_pos_params(self):
        for name, p in self.core.pretrained.named_parameters():
            if "pos_embed" in name:
                yield p

    def get_enc_params_except_rel_pos(self):
        for name, p in self.core.pretrained.named_parameters():
            if "pos_embed" not in name:
                yield p

    def freeze_encoder(self, freeze_rel_pos=False):
        if freeze_rel_pos:
            for p in self.core.pretrained.parameters():
                p.requires_grad = False
        else:
            for p in self.get_enc_params_except_rel_pos():
                p.requires_grad = False
        return self

    def attach_hooks(self, midas):
        if len(self.handles) > 0:
            self.remove_hooks()
        if "out_conv" in self.layer_names:
            self.handles.append(list(midas.depth_head.scratch.output_conv2.children())[
                                1].register_forward_hook(get_activation("out_conv", self.core_out)))
        if "r4" in self.layer_names:
            self.handles.append(midas.depth_head.scratch.refinenet4.register_forward_hook(
                get_activation("r4", self.core_out)))
        if "r3" in self.layer_names:
            self.handles.append(midas.depth_head.scratch.refinenet3.register_forward_hook(
                get_activation("r3", self.core_out)))
        if "r2" in self.layer_names:
            self.handles.append(midas.depth_head.scratch.refinenet2.register_forward_hook(
                get_activation("r2", self.core_out)))
        if "r1" in self.layer_names:
            self.handles.append(midas.depth_head.scratch.refinenet1.register_forward_hook(
                get_activation("r1", self.core_out)))
        if "l4_rn" in self.layer_names:
            self.handles.append(midas.depth_head.scratch.layer4_rn.register_forward_hook(
                get_activation("l4_rn", self.core_out)))

        return self

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        return self

    def __del__(self):
        self.remove_hooks()

    def set_output_channels(self):
        self.output_channels = [256, 256, 256, 256, 256]

    @staticmethod
    def build(midas_model_type="dinov2_large", train_midas=False, use_pretrained_midas=True, fetch_features=False, freeze_bn=True, force_keep_ar=False, force_reload=False, **kwargs):
        if "img_size" in kwargs:
            kwargs = DepthAnythingCore.parse_img_size(kwargs)
        img_size = kwargs.pop("img_size", [384, 384])
        
        depth_anything = DPT_DINOv2(out_channels=[256, 512, 1024, 1024], use_clstoken=False)
        
        state_dict = torch.load('./checkpoints/depth_anything_vitl14.pth', map_location='cpu')
        depth_anything.load_state_dict(state_dict)
        
        kwargs.update({'keep_aspect_ratio': force_keep_ar})
        
        depth_anything_core = DepthAnythingCore(depth_anything, trainable=train_midas, fetch_features=fetch_features,
                               freeze_bn=freeze_bn, img_size=img_size, **kwargs)
        
        depth_anything_core.set_output_channels()
        return depth_anything_core

    @staticmethod
    def parse_img_size(config):
        assert 'img_size' in config
        if isinstance(config['img_size'], str):
            assert "," in config['img_size'], "img_size should be a string with comma separated img_size=H,W"
            config['img_size'] = list(map(int, config['img_size'].split(",")))
            assert len(
                config['img_size']) == 2, "img_size should be a string with comma separated img_size=H,W"
        elif isinstance(config['img_size'], int):
            config['img_size'] = [config['img_size'], config['img_size']]
        else:
            assert isinstance(config['img_size'], list) and len(
                config['img_size']) == 2, "img_size should be a list of H,W"
        return config


class DepthAnythingCore_Dual(nn.Module):
    def __init__(self, midas_dual, trainable=False, fetch_features=True, layer_names=('out_conv', 'l4_rn', 'r4', 'r3', 'r2', 'r1'), freeze_bn=False, keep_aspect_ratio=True,
                 img_size=384, fusion_block=None, **kwargs):
        """
        Args:
            midas (torch.nn.Module): Midas model.
            trainable (bool, optional): Train midas model. Defaults to False.
            fetch_features (bool, optional): Extract multi-scale features. Defaults to True.
            layer_names (tuple, optional): Layers used for feature extraction. Order = (head output features, last layer features, ...decoder features). Defaults to ('out_conv', 'l4_rn', 'r4', 'r3', 'r2', 'r1').
            freeze_bn (bool, optional): Freeze BatchNorm. Generally results in better finetuning performance. Defaults to False.
            keep_aspect_ratio (bool, optional): Keep the aspect ratio of input images while resizing. Defaults to True.
            img_size (int, tuple, optional): Input resolution. Defaults to 384.
        """
        super().__init__()
        self.core = midas_dual
        self.output_channels = None
        self.core_out = {}

        self.fusion_block = fusion_block
        self.core_out_extra = {}

        self.trainable = trainable
        self.fetch_features = fetch_features
        # midas.scratch.output_conv = nn.Identity()
        self.handles = []
        # self.layer_names = ['out_conv','l4_rn', 'r4', 'r3', 'r2', 'r1']
        self.layer_names = layer_names

        self.set_trainable(trainable)
        self.set_fetch_features(fetch_features)
        self.prep_dual = PrepForMidas_Dual(keep_aspect_ratio=keep_aspect_ratio, img_size=img_size, do_resize=kwargs.get('do_resize', True))

        if freeze_bn:
            self.freeze_bn()


    def set_trainable(self, trainable):
        self.trainable = trainable
        if trainable:
            self.unfreeze()
        else:
            self.freeze()
        return self

    def set_fetch_features(self, fetch_features):
        self.fetch_features = fetch_features
        if fetch_features:
            if len(self.handles) == 0:
                self.attach_hooks(self.core)
        else:
            self.remove_hooks()
        return self

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.trainable = False
        return self

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self.trainable = True
        return self

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x, y, denorm=False, return_rel_depth=False):
        """
        x, y: input thermal images, and rgb images respectively
        """
        # print('input to midas_dual:', x.shape)
        with torch.no_grad():
            if denorm:
                x = denormalize(x)
                y = denormalize(y)
            x, y = self.prep_dual(x, y) #NOTE: x is thermal, y is rgb!!!
        # print("======================== for debug1, in DepthAnythingCore_Dual, x, y shape: ", x.shape, y.shape)

        with torch.set_grad_enabled(self.trainable):
            rel_depth, rel_depth_extra, feat_dino, feat_dino_extra = self.core(x, y) # feat_dino [B, h//14*w//14, 1024]
            if not self.fetch_features:
                return rel_depth, rel_depth_extra

        out = [self.core_out[k] for k in self.layer_names]
        out_extra = [self.core_out_extra[k] for k in self.layer_names]

        feat_h, feat_w, feat_ch = x.shape[-2]//14, x.shape[-1]//14, feat_dino.shape[-1]
        out.append(feat_dino.permute(0, 2, 1).reshape((feat_dino.shape[0], feat_ch, feat_h, feat_w)))
        out_extra.append(feat_dino_extra.permute(0, 2, 1).reshape((feat_dino.shape[0], feat_ch, feat_h, feat_w)))

        # ### For debug, pint the shape of the outfeatures
        # print("--------------------------------")
        # # For input img shape [1, C, 196, 602]; The features in out: [1, 32, 196, 602], [1, 256, 7, 22], [1, 256, 14, 43], [1, 256, 28, 86], [1, 256, 56, 172], [1, 256, 112, 344]
        # for o in out:
        #     print("Shape of the output feature: ", o.shape)
        # print("--------------------------------")

        if return_rel_depth:
            return rel_depth, rel_depth_extra, out, out_extra
        return out, out_extra


    def get_rel_pos_params(self):
        for name, p in self.core.pretrained.named_parameters():
            if "pos_embed" in name:
                yield p
        for name_extra, p_extra in self.core.pretrained_extra.named_parameters():
            if "pos_embed" in name_extra:
                yield p_extra

    def get_enc_params_except_rel_pos(self):
        for name, p in self.core.pretrained.named_parameters():
            if "pos_embed" not in name:
                yield p
        for name_extra, p_extra in self.core.pretrained_extra.named_parameters():
            if "pos_embed" not in name_extra:
                yield p_extra

    def freeze_encoder(self, freeze_rel_pos=False):
        if freeze_rel_pos:
            for p in self.core.pretrained.parameters():
                p.requires_grad = False
            for p in self.core.pretrained_extra.parameters():
                p.requires_grad = False
        else:
            for p in self.get_enc_params_except_rel_pos():
                p.requires_grad = False
        return self

    def attach_hooks(self, midas_dual):
        if len(self.handles) > 0:
            self.remove_hooks()
        ### For the first branch
        if "out_conv" in self.layer_names:
            self.handles.append(list(midas_dual.depth_head.scratch.output_conv2.children())[
                                1].register_forward_hook(get_activation("out_conv", self.core_out)))
        if "r4" in self.layer_names:
            self.handles.append(midas_dual.depth_head.scratch.refinenet4.register_forward_hook(
                get_activation("r4", self.core_out)))
        if "r3" in self.layer_names:
            self.handles.append(midas_dual.depth_head.scratch.refinenet3.register_forward_hook(
                get_activation("r3", self.core_out)))
        if "r2" in self.layer_names:
            self.handles.append(midas_dual.depth_head.scratch.refinenet2.register_forward_hook(
                get_activation("r2", self.core_out)))
        if "r1" in self.layer_names:
            self.handles.append(midas_dual.depth_head.scratch.refinenet1.register_forward_hook(
                get_activation("r1", self.core_out)))
        if "l4_rn" in self.layer_names:
            self.handles.append(midas_dual.depth_head.scratch.layer4_rn.register_forward_hook(
                get_activation("l4_rn", self.core_out)))

        ### For the second branch
        if "out_conv" in self.layer_names:
            self.handles.append(list(midas_dual.depth_head_extra.scratch.output_conv2.children())[
                                1].register_forward_hook(get_activation("out_conv", self.core_out_extra)))
        if "r4" in self.layer_names:
            self.handles.append(midas_dual.depth_head_extra.scratch.refinenet4.register_forward_hook(
                get_activation("r4", self.core_out_extra)))
        if "r3" in self.layer_names:
            self.handles.append(midas_dual.depth_head_extra.scratch.refinenet3.register_forward_hook(
                get_activation("r3", self.core_out_extra)))
        if "r2" in self.layer_names:
            self.handles.append(midas_dual.depth_head_extra.scratch.refinenet2.register_forward_hook(
                get_activation("r2", self.core_out_extra)))
        if "r1" in self.layer_names:
            self.handles.append(midas_dual.depth_head_extra.scratch.refinenet1.register_forward_hook(
                get_activation("r1", self.core_out_extra)))
        if "l4_rn" in self.layer_names:
            self.handles.append(midas_dual.depth_head_extra.scratch.layer4_rn.register_forward_hook(
                get_activation("l4_rn", self.core_out_extra)))

        return self

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        return self

    def __del__(self):
        self.remove_hooks()

    def set_output_channels(self):
        self.output_channels = [256, 256, 256, 256, 256]

    def set_extras(self):
        self.core.set_extras()
        # if self.fetch_features:
        #     self.attach_hooks(self.core)

    def fix_weights_extras(self):
        self.core.fix_weights_extras()

    def set_attach_hooks(self):
        if self.fetch_features:
            self.attach_hooks(self.core)

    @staticmethod
    def build(midas_model_type="dinov2_large", train_midas=False, use_pretrained_midas=True, fetch_features=False, freeze_bn=True, force_keep_ar=False, force_reload=False, fusion_block=None, **kwargs):
        if "img_size" in kwargs:
            kwargs = DepthAnythingCore_Dual.parse_img_size(kwargs)
        img_size = kwargs.pop("img_size", [384, 384])

        depth_anything_dual = DPT_DINOv2_dual(out_channels=[256, 512, 1024, 1024], use_clstoken=False, fusion_block=fusion_block)
        state_dict = torch.load('./checkpoints/depth_anything_vitl14.pth', map_location='cpu')
        depth_anything_dual.load_state_dict(state_dict, strict=False)

        if 'depth_head_extra.projects.0.weight' in state_dict:
            print("---------- Weights for 'pretrained_extra, depth_head_extra, etc' are present.")
        else:
            print("---------- Weights for 'pretrained_extra, depth_head_extra, etc' are not present. Calling set_extras()! ----------")
            depth_anything_dual.set_extras()  # Copy weights to the second branch, except the fusion_block

        kwargs.update({'keep_aspect_ratio': force_keep_ar})
        depth_anything_core_dual = DepthAnythingCore_Dual(depth_anything_dual, trainable=train_midas, fetch_features=fetch_features,
                               freeze_bn=freeze_bn, img_size=img_size, **kwargs)

        depth_anything_core_dual.set_output_channels()
        return depth_anything_core_dual

    @staticmethod
    def parse_img_size(config):
        assert 'img_size' in config
        if isinstance(config['img_size'], str):
            assert "," in config['img_size'], "img_size should be a string with comma separated img_size=H,W"
            config['img_size'] = list(map(int, config['img_size'].split(",")))
            assert len(
                config['img_size']) == 2, "img_size should be a string with comma separated img_size=H,W"
        elif isinstance(config['img_size'], int):
            config['img_size'] = [config['img_size'], config['img_size']]
        else:
            assert isinstance(config['img_size'], list) and len(
                config['img_size']) == 2, "img_size should be a list of H,W"
        return config

nchannels2models = {
    tuple([256]*5): ["DPT_BEiT_L_384", "DPT_BEiT_L_512", "DPT_BEiT_B_384", "DPT_SwinV2_L_384", "DPT_SwinV2_B_384", "DPT_SwinV2_T_256", "DPT_Large", "DPT_Hybrid"],
    (512, 256, 128, 64, 64): ["MiDaS_small"]
}

# Model name to number of output channels
MIDAS_SETTINGS = {m: k for k, v in nchannels2models.items()
                  for m in v
                  }
