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
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np

from zoedepth.utils.depth_utils import colored_depthmap
import cv2
import math


KEY_OUTPUT = 'metric_depth'


def extract_key(prediction, key):
    if isinstance(prediction, dict):
        return prediction[key]
    return prediction

def get_mask_exlude_top20(abs_diff, top_percent=0.2):
    with torch.no_grad():
        # Flatten the errors for each image and sort them
        downsampled_abs_diff = abs_diff[:, :, ::10, ::10]
        B, _, H, W = downsampled_abs_diff.clone().shape
        abs_diff_flat = downsampled_abs_diff.reshape(B, -1)  # Shape: (B, H*W)
        sorted_diff, _ = torch.sort(abs_diff_flat, dim=1)  # Shape: (B, H*W)

        # Determine the threshold value for the top 20% errors
        threshold_index = int((1 - top_percent) * H * W)
        threshold_values = sorted_diff[:, threshold_index]  # Shape: (B,)

        # Reshape the threshold values to match the original tensor shape
        threshold_values = threshold_values.view(B, 1, 1, 1)  # Shape: (B, 1, 1, 1)

        # Create a mask to exclude the top 20% errors
        mask = (abs_diff <= threshold_values).detach()  # Shape: (B, 1, H, W)
    return mask.detach()


# Main loss function used for ZoeDepth. Copy/paste from AdaBins repo (https://github.com/shariqfarooq123/AdaBins/blob/0952d91e9e762be310bb4cd055cbfe2448c0ce20/loss.py#L7)
class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""
    def __init__(self, beta=0.15):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False, bexlude_top20=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True) # 'bilinear'
            intr_input = input
        else:
            intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)

        ## Rocky: aug the mask to exclude the ones with top 20% errors
        if bexlude_top20:
            # Calculate the absolute differences (errors)
            abs_diff = torch.abs(target - input)  # Shape: (B, 1, H, W)
            mask_exclude_top20 = get_mask_exlude_top20(abs_diff, top_percent=0.20)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            if bexlude_top20:
                mask = mask & mask_exclude_top20

            input = input[mask]
            target = target[mask]
        elif bexlude_top20:
            mask = mask_exclude_top20
            input = input[mask]
            target = target[mask]


        with amp.autocast(enabled=False):  # amp causes NaNs in this loss function
            alpha = 1e-7
            g = torch.log(input + alpha) - torch.log(target + alpha)

            # n, c, h, w = g.shape
            # norm = 1/(h*w)
            # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2
            # Rocky: The above is exactly scal invariant loss from Eigen paper: Depth Map Prediction from a Single Image
            # using a Multi-Scale Deep Network, when beta=0.85 is changed to 1.0

            Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2) # This is not scale invariant.

            loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print("Nan SILog loss")
            print("input:", input.shape)
            print("target:", target.shape)
            print("G", torch.sum(torch.isnan(g)))
            print("Input min max", torch.min(input), torch.max(input))
            print("Target min max", torch.min(target), torch.max(target))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))

        if not return_interpolated:
            return loss

        return loss, intr_input


def grad(x):
    # x.shape : n, c, h, w
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    mag = diff_x**2 + diff_y**2
    # angle_ratio
    angle = torch.atan(diff_y / (diff_x + 1e-10))
    return mag, angle


def grad_mask(mask):
    return mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]


class GradL1Loss(nn.Module):
    """Gradient loss"""
    def __init__(self):
        super(GradL1Loss, self).__init__()
        self.name = 'GradL1'

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        grad_gt = grad(target)
        grad_pred = grad(input)
        mask_g = grad_mask(mask)

        loss = nn.functional.l1_loss(grad_pred[0][mask_g], grad_gt[0][mask_g])
        loss = loss + \
            nn.functional.l1_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])
        if not return_interpolated:
            return loss
        return loss, intr_input

def get_disparity_smooth_loss(input_depth, input_img, bexlude_top20=False, binvse_input=False):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness; both input_depth, img, mask are with shape BCHW
    """
    depth = extract_key(input_depth, KEY_OUTPUT)
    disp = 1.0 / (depth + 1e-6) if binvse_input else depth
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    if input_img.shape[-1] != input_depth.shape[-1]:
        img = nn.functional.interpolate(
            input_img, input_depth.shape[-2:], mode='bilinear', align_corners=True)
    else:
        img = input_img


    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    if bexlude_top20:
        mask_x = get_mask_exlude_top20(grad_img_x)
        grad_disp_x = grad_disp_x[mask_x]

        mask_y = get_mask_exlude_top20(grad_img_y)
        grad_disp_y = grad_disp_y[mask_y]

    return grad_disp_x.mean() + grad_disp_y.mean()


class L1Loss(nn.Module):
    """Gradient loss"""
    def __init__(self):
        super(L1Loss, self).__init__()
        self.name = 'L1_discrepancy'

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input
        loss = 0.
        if mask is None:
            loss = nn.functional.l1_loss(intr_input, target)
        else:
            loss = nn.functional.l1_loss(intr_input[mask], target[mask])

        if not return_interpolated:
            return loss
        return loss, intr_input

### TODO: Check its validity
def ssim(A, B, valid_mask=None, imask=None):
    C1 = 0.01**2
    C2 = 0.03**2
    mu_A = F.avg_pool2d(A, 3)
    mu_B = F.avg_pool2d(B, 3)
    sigma_A  = F.avg_pool2d(A**2, 3)-mu_A**2
    sigma_B  = F.avg_pool2d(B**2, 3)-mu_B**2
    sigma_AB = F.avg_pool2d(A*B , 3)-mu_A*mu_B
    numer = (2*mu_A*mu_B+C1)*(2*sigma_AB+C2)
    denom = (mu_A**2+mu_B**2+C1)*(sigma_A+sigma_B+C2)
    score = numer / denom
    sim = torch.clamp((1-score)/2, 0, 1)
    if imask is not None:
        sim = sim * imask
    if valid_mask is not None:
        vmask = F.max_pool2d(valid_mask.float(), 3).bool()
        sim = sim[vmask]
    return sim.mean()


class Nll_loss(nn.Module):
    """Negative log likelihood of Laplassian loss"""
    def __init__(self):
        super(Nll_loss, self).__init__()
        self.name = 'Nll'

    def forward(self, input, target, varinv, mask=None, interpolate=True, return_interpolated=False, beta_logwar_weight=0.002, ignore_bigdiff=True):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if varinv.shape[-1] != target.shape[-1] and interpolate:
            varinv = nn.functional.interpolate(
                varinv, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_varinv = varinv
        else:
            intr_varinv = varinv

        # min_varinv = 1e-6
        # cramped_varinv = torch.where(intr_varinv > min_varinv, intr_varinv, torch.tensor(min_varinv).cuda()).to(torch.float)
        cramped_varinv = intr_varinv

        diff = torch.abs(target - intr_input)
        if ignore_bigdiff:
            mask_exclude_top20 = get_mask_exlude_top20(diff, top_percent=0.20)
            mask = (mask & mask_exclude_top20) if mask is not None else mask_exclude_top20
            # random_mask = torch.rand_like(diff) < 0.3
            # mask = mask & random_mask
            # print("############ For debug, mask.sum().item(), mean std median max min of the diff: ", mask.sum().item(),diff[mask].mean().item(), diff[mask].std().item(), diff[mask].median().item(), diff[mask].max().item(), diff[mask].min().item())

        if mask is None:
            loss = ((diff)*cramped_varinv).mean() - beta_logwar_weight * torch.log(cramped_varinv).mean()
        else:
            mask = mask.detach()
            loss = ((diff)*cramped_varinv)[mask].mean() - beta_logwar_weight * torch.log(cramped_varinv[mask]).mean()

        if torch.isnan(loss):
            print("Nan Nll_loss loss")
            print("input:", input.shape)
            print("target:", target.shape)
            print("cramped_varinv: ", cramped_varinv.shape)
            print("Input min max", torch.min(input), torch.max(input))
            print("Target min max", torch.min(target), torch.max(target))
            print("cramped_varinv min max", torch.min(cramped_varinv), torch.max(cramped_varinv))
            print("mask.sum(), make.shape: ", mask.sum(), mask.shape)
            print("loss", torch.isnan(loss))

        if not return_interpolated:
            return loss
        return loss, intr_input

class L1Loss_window(nn.Module):
    """Gradient loss"""
    def __init__(self):
        super(L1Loss_window, self).__init__()
        self.name = 'L1_window_discrepancy'

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False, window_size=0, brelative_mask=True, error_threshold = 1.0):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input
        loss = 0.
        B, C, H, W = intr_input.shape

        if window_size > 1:
            # Calculate padding size
            pad = window_size // 2
            # Pad the target to keep the windows centered
            padded_target = F.pad(target, (pad, pad, pad, pad), mode='constant', value=0)

            # Unfold the padded target to create sliding windows
            # unfolded_target shape: B x C x (H+2*pad-window_size+1) x (W+2*pad-window_size+1) x window_size x window_size
            unfolded_target = padded_target.unfold(2, window_size, 1).unfold(3, window_size, 1)

            # Unfold should match the dimensions of the original intr_input
            unfolded_target = unfolded_target[:, :, :H, :W, :, :]

            # Flatten the last two dimensions (window_size x window_size)
            # unfolded_target shape: B x C x H x W x (window_size*window_size)
            unfolded_target = unfolded_target.contiguous().view(B, C, H, W, -1)

            # Expand intr_input to match the shape of unfolded_target
            # expanded_intr_input shape: B x C x H x W x 1
            expanded_intr_input = intr_input.unsqueeze(-1)

            # Compute the L1 loss between intr_input and each window in the unfolded_target
            # l1_losses shape: B x C x H x W x (window_size*window_size)
            l1_losses = torch.abs(expanded_intr_input - unfolded_target)

            # Find the minimum L1 loss for each position
            # min_l1_losses shape: B x C x H x W
            min_l1_losses, _ = l1_losses.min(dim=-1)
        else:    # pixel loss
            min_l1_losses = torch.abs(intr_input - target)



        mask_final = mask
        if brelative_mask:
            relative_mask =  (torch.maximum(min_l1_losses/(target+1e-3), min_l1_losses/(input+1e-3))  < 0.25) & (min_l1_losses < error_threshold) # B x C x H x W
            mask_final = relative_mask if mask is None else (mask & relative_mask)

        if mask_final is None:
            loss = torch.mean(min_l1_losses)
        else:
            loss = torch.mean(min_l1_losses[mask_final])

        # ### For debug, vislize the input and supervision
        # depth_error = min_l1_losses[0]
        # depth_error[mask_final[0] == 0] = 0
        # print("---- Debug only, min_l1_losses[[mask_final]], mean(), std(), median(), max(), min()", min_l1_losses[mask_final].mean().item(),
        #       min_l1_losses[mask_final].std().item(), min_l1_losses[mask_final].median().item(), min_l1_losses[mask_final].max().item(),
        #       min_l1_losses[mask_final].min().item())

        # vis_depth = colored_depthmap(intr_input[0].squeeze().detach().cpu().numpy(), 0, 7).astype(np.uint8)
        # vis_depth1 = colored_depthmap(target[0].squeeze().detach().cpu().numpy(), 0, 7).astype(np.uint8)
        # vis_depth_error = colored_depthmap(depth_error.squeeze().detach().cpu().numpy(), 0, 1).astype(np.uint8)
        # vis_check_warp = np.hstack((vis_depth, vis_depth1, vis_depth_error))
        # cv2.imshow("inputdepth-targetdepth-vis_depth_error", vis_check_warp)
        # cv2.waitKey()

        if not return_interpolated:
            return loss
        return loss, intr_input

class OrdinalRegressionLoss(object):

    def __init__(self, ord_num, beta, discretization="SID"):
        self.ord_num = ord_num
        self.beta = beta
        self.discretization = discretization

    def _create_ord_label(self, gt):
        N,one, H, W = gt.shape
        # print("gt shape:", gt.shape)

        ord_c0 = torch.ones(N, self.ord_num, H, W).to(gt.device)
        if self.discretization == "SID":
            label = self.ord_num * torch.log(gt) / np.log(self.beta)
        else:
            label = self.ord_num * (gt - 1.0) / (self.beta - 1.0)
        label = label.long()
        mask = torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False) \
            .view(1, self.ord_num, 1, 1).to(gt.device)
        mask = mask.repeat(N, 1, H, W).contiguous().long()
        mask = (mask > label)
        ord_c0[mask] = 0
        ord_c1 = 1 - ord_c0
        # implementation according to the paper.
        # ord_label = torch.ones(N, self.ord_num * 2, H, W).to(gt.device)
        # ord_label[:, 0::2, :, :] = ord_c0
        # ord_label[:, 1::2, :, :] = ord_c1
        # reimplementation for fast speed.
        ord_label = torch.cat((ord_c0, ord_c1), dim=1)
        return ord_label, mask

    def __call__(self, prob, gt):
        """
        :param prob: ordinal regression probability, N x 2*Ord Num x H x W, torch.Tensor
        :param gt: depth ground truth, NXHxW, torch.Tensor
        :return: loss: loss value, torch.float
        """
        # N, C, H, W = prob.shape
        valid_mask = gt > 0.
        ord_label, mask = self._create_ord_label(gt)
        # print("prob shape: {}, ord label shape: {}".format(prob.shape, ord_label.shape))
        entropy = -prob * ord_label
        loss = torch.sum(entropy, dim=1)[valid_mask.squeeze(1)]
        return loss.mean()


class DiscreteNLLLoss(nn.Module):
    """Cross entropy loss"""
    def __init__(self, min_depth=1e-3, max_depth=10, depth_bins=64):
        super(DiscreteNLLLoss, self).__init__()
        self.name = 'CrossEntropy'
        self.ignore_index = -(depth_bins + 1)
        # self._loss_func = nn.NLLLoss(ignore_index=self.ignore_index)
        self._loss_func = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_bins = depth_bins
        self.alpha = 1
        self.zeta = 1 - min_depth
        self.beta = max_depth + self.zeta

    def quantize_depth(self, depth):
        # depth : N1HW
        # output : NCHW

        # Quantize depth log-uniformly on [1, self.beta] into self.depth_bins bins
        depth = torch.log(depth / self.alpha) / np.log(self.beta / self.alpha)
        depth = depth * (self.depth_bins - 1)
        depth = torch.round(depth) 
        depth = depth.long()
        return depth
        

    
    def _dequantize_depth(self, depth):
        """
        Inverse of quantization
        depth : NCHW -> N1HW
        """
        # Get the center of the bin




    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        # assert torch.all(input <= 0), "Input should be negative"

        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        # assert torch.all(input)<=1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        target = self.quantize_depth(target)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            # Set the mask to ignore_index
            mask = mask.long()
            input = input * mask + (1 - mask) * self.ignore_index
            target = target * mask + (1 - mask) * self.ignore_index

        

        input = input.flatten(2)  # N, nbins, H*W
        target = target.flatten(1)  # N, H*W
        loss = self._loss_func(input, target)

        if not return_interpolated:
            return loss
        return loss, intr_input
    



def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1
class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SSILoss"

    def forward(self, prediction, target, mask, interpolate=True, return_interpolated=False):
        
        if prediction.shape[-1] != target.shape[-1] and interpolate:
            prediction = nn.functional.interpolate(prediction, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = prediction
        else:
            intr_input = prediction


        prediction, target, mask = prediction.squeeze(), target.squeeze(), mask.squeeze()
        assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

        scale, shift = compute_scale_and_shift(prediction, target, mask)

        scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        loss = nn.functional.l1_loss(scaled_prediction[mask], target[mask])
        if not return_interpolated:
            return loss
        return loss, intr_input




if __name__ == '__main__':
    # Tests for DiscreteNLLLoss
    celoss = DiscreteNLLLoss()
    print(celoss(torch.rand(4, 64, 26, 32)*10, torch.rand(4, 1, 26, 32)*10, ))

    d = torch.Tensor([6.59, 3.8, 10.0])
    print(celoss.dequantize_depth(celoss.quantize_depth(d)))
