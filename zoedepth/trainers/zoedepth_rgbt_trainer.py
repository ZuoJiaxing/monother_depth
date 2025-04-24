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
import torch.cuda.amp as amp
import torch.nn as nn

import torch.nn.functional as F
from torch.nn.functional import grid_sample

from zoedepth.trainers.loss import GradL1Loss, SILogLoss, L1Loss, L1Loss_window, get_disparity_smooth_loss, Nll_loss
from zoedepth.utils.config import DATASETS_CONFIG
from zoedepth.utils.misc import compute_metrics
from zoedepth.data.preprocess import get_black_border
from zoedepth.utils.depth_utils import util_add_row,colored_depthmap, remap_pixel_coordinates, generate_depth_map_with_occlusion

from .base_trainer import BaseTrainer
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Trainer(BaseTrainer):
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        super().__init__(config, model, train_loader,
                         test_loader=test_loader, device=device)
        self.device = device
        self.silog_loss = SILogLoss()
        self.grad_loss = GradL1Loss()
        self.l1_loss = L1Loss()
        self.l1_loss_window = L1Loss_window()
        self.nll_loss = Nll_loss()
        self.scaler = amp.GradScaler(enabled=self.config.use_amp)

    def train_on_batch(self, batch, train_step):
        """
        Expects a batch of images and depth as input
        batch["image"].shape : batch_size, c, h, w; thermal image
        batch["depth"].shape : batch_size, 1, h, w; thermal depth
        batch["rgb_image"].shape : batch_size, c, h, w; rgb image
        batch["depth_image"].shape : batch_size, 1, h, w; thermal image
        batch["tpix_in_rgb"].shape : batch_size, 2, h, w; the coordinates of thermal pixels in rgb image
        """

        images, depths_gt = batch['image'].to(self.device), batch['depth'].to(self.device) # thermal image and its depth
        dataset = batch['dataset'][0]

        images_rgb, depths_gt_rgb = batch['rgb_image'].to(self.device), batch['rgb_depth'].to(self.device) # rgb image

        b, c, h, w = images.size() # The images sizes are not resized yet.
        mask = batch["mask"].to(self.device).to(torch.bool)
        mask_rgb = batch["rgb_mask"].to(self.device).to(torch.bool)
        losses = {}



        # ### For debug, vislize the input and supervision
        # vis_depth = colored_depthmap(depths_gt[0].squeeze().cpu().numpy(), 0, 7).astype(np.uint8)
        # vis_img = 255.0 * images[0].squeeze().cpu().numpy()
        # vis_img = vis_img.transpose(1, 2, 0).astype(np.uint8)
        # vis_img_rgb = 255.0 * images_rgb[0].squeeze().cpu().numpy()
        # vis_img_rgb = vis_img_rgb.transpose(1, 2, 0).astype(np.uint8)
        # vis_depth_rgb = colored_depthmap(depths_gt_rgb[0].squeeze().cpu().numpy(), 0, 7).astype(np.uint8)
        #
        # vis_check_warp = np.hstack((vis_img, vis_depth, vis_img_rgb, vis_depth_rgb))
        # cv2.imshow("input: thermal-depth-rgb-visdepth", vis_check_warp)
        # cv2.waitKey()

        model_return_probs = False
        if self.config.w_rgbt_kld > 0:
            model_return_probs = True

        bcompute_nll_loss = self.config.w_rgbt_nll > 0

        with ((amp.autocast(enabled=self.config.use_amp))):
            if not bcompute_nll_loss:
                output, output_extra=self.model(images, images_rgb, return_probs=model_return_probs) # The images will be resized in the model.
            else:
                thermal_intrinsics = batch['intrinsics'].to(self.device) if 'intrinsics' in batch else batch[
                    'proj_matrix'].to(self.device)
                rgb_intrinsics = batch['rgb_intrinsics'].to(self.device) if 'rgb_intrinsics' in batch else batch[
                    'rgb_proj_matrix'].to(self.device)
                pose_t2rgb = batch['pose_t2rgb'].to(self.device)
                output, output_extra = self.model(images, images_rgb, return_probs=model_return_probs, nll_loss=bcompute_nll_loss, intrinsics_x=thermal_intrinsics, intrinsics_y=rgb_intrinsics, extrinsincs_x2y=pose_t2rgb)  # The images will be resized in the model.

            pred_depths = output['metric_depth'] # (B, 1, H, W)
            pred_depths_rgb = output_extra['metric_depth'] # (B, 1, H, W)
            pred_w, pred_h = pred_depths.shape[-1], pred_depths.shape[-2]

            loss = 0.0
            pred_depths_rgb_intp = None
            pred_depths_intp = None
            if self.config.w_si > 0: # RGB depth prediction loss
                l_si, pred_depths_rgb_intp = self.silog_loss(
                    pred_depths_rgb, depths_gt_rgb, mask=mask_rgb, interpolate=True, return_interpolated=True, bexlude_top20=True) # Loss for the rgb depth prediction
                loss = loss + self.config.w_si * l_si
                losses[self.silog_loss.name+"_rgb"] = l_si

            if self.config.w_si_thermal > 0: # Thermal depth prediction loss
                l_si, pred_depths_intp = self.silog_loss(
                    pred_depths, depths_gt, mask=mask, interpolate=True, return_interpolated=True, bexlude_top20=True) # Loss for the rgb depth prediction
                loss = loss + self.config.w_si_thermal * l_si
                losses[self.silog_loss.name+"_thermal"] = l_si

            # # Visualize the depth and images
            # print("---------------- images.shape, images_rgb.shape: ", images.shape, images_rgb.shape)
            # intep_pred_depths_rgb = nn.functional.interpolate(pred_depths_rgb.detach(), (h, w), mode='bilinear', align_corners=True)
            # vis_pred_depths_rgb = colored_depthmap(intep_pred_depths_rgb[0].squeeze().cpu().numpy(), 0, self.config.max_depth).astype(np.uint8)
            # intep_pred_depths = nn.functional.interpolate(pred_depths.detach(), (h, w), mode='bilinear', align_corners=True)
            # vis_pred_depths = colored_depthmap(intep_pred_depths[0].squeeze().cpu().numpy(), 0, self.config.max_depth).astype(np.uint8)
            # vis_gt_depths_rgb = colored_depthmap(depths_gt_rgb[0].squeeze().cpu().numpy(), 0, self.config.max_depth).astype(np.uint8)
            # vis_gt_depths = colored_depthmap(depths_gt[0].squeeze().cpu().numpy(), 0, self.config.max_depth).astype(np.uint8)
            # # vis_img = F.interpolate(images_rgb, size=(pred_h, pred_w), mode='nearest')
            # vis_img = 255.0 * images[0].squeeze().cpu().numpy()
            # vis_img = vis_img.transpose(1, 2, 0).astype(np.uint8)
            # vis_img_rgb = 255.0 * images_rgb[0].squeeze().cpu().numpy()
            # vis_img_rgb = vis_img_rgb.transpose(1, 2, 0).astype(np.uint8)
            # vis_check_warp = np.hstack((vis_img_rgb, vis_pred_depths_rgb, vis_gt_depths_rgb, vis_img, vis_pred_depths, vis_gt_depths))
            # cv2.imshow("rgb-pred-gt-ther-pred-gt", vis_check_warp)
            # cv2.waitKey(10)



            # ### Debug: Calculate the mask when computing the consistency loss.
            # ## Warp the thermal depth to the rgb depth
            # thermal_intrinsics = batch['intrinsics'].to(self.device) if 'intrinsics' in batch else batch['proj_matrix'].to(self.device)
            # rgb_intrinsics = batch['rgb_intrinsics'].to(self.device) if 'rgb_intrinsics' in batch else batch['rgb_proj_matrix'].to(self.device)
            # # pose_t2rgb = batch['pose_t2rgb'].to(self.device)
            # # Refer to https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6
            # # We should apply identity matrix, since P1 and P2 projection matrces, and both project points from coordiated of rectified cam1.
            # pose_t2rgb = torch.eye(4).unsqueeze(0).repeat(b, 1, 1).to(self.device)
            #
            # # print("=============== Shape of thermal_intrinsics, rgb_intrinsics, pose_t2rgb: ", thermal_intrinsics.shape, rgb_intrinsics.shape, pose_t2rgb.shape)
            # # print("=============== thermal_intrinsics, rgb_intrinsics, pose_t2rgb: ", thermal_intrinsics, rgb_intrinsics, pose_t2rgb)
            # # print("=============== pred_depths.shape", pred_depths.shape)
            #
            # pred_h, pred_w = pred_depths.shape[-2], pred_depths.shape[-1]
            # scale_h, scale_w = pred_h/h, pred_w/w
            # thermal_intrinsics[:, 0, :] = thermal_intrinsics[:, 0, :] * scale_w
            # thermal_intrinsics[:, 1, :] = thermal_intrinsics[:, 1, :] * scale_h
            # rgb_intrinsics[:, 0, :] = rgb_intrinsics[:, 0, :] * scale_w
            # rgb_intrinsics[:, 1, :] = rgb_intrinsics[:, 1, :] * scale_h
            #
            # ### Warp the thermal depth to rgb depth
            # pred_depths_tmp = F.interpolate(depths_gt, size=(pred_h, pred_w), mode='nearest').permute(0,2,3,1).contiguous()  # ERROR: for debug only
            # rgb_depths_tmp = batch['rgb_depth'].to(self.device)
            # rgb_depths_tmp = F.interpolate(rgb_depths_tmp, size=(pred_h, pred_w), mode='nearest')  # ERROR: for debug only
            # tpix_in_rgb, tdepth_in_rgb = remap_pixel_coordinates(pred_depths_tmp, thermal_intrinsics, rgb_intrinsics, pose_cam1tocam2=None) # pose_cam1tocam2
            #
            # # First way to calculate the depth discrepancy
            # tdepthmap_in_rgb = generate_depth_map_with_occlusion(tpix_in_rgb, tdepth_in_rgb, pred_h, pred_w) # (B, 1, H, W)
            # valid_mask_tmp = (tdepthmap_in_rgb>0.1) & (rgb_depths_tmp>0.1) & (tdepthmap_in_rgb < self.config.max_depth) & (rgb_depths_tmp < self.config.max_depth)
            # diff = (rgb_depths_tmp - tdepthmap_in_rgb).abs()
            # print("================== for debug, checking the warp, valid_mask_tmp and error-mean, error-median: ", torch.sum(valid_mask_tmp).item(), torch.mean(diff[valid_mask_tmp]).item(),  torch.median(diff[valid_mask_tmp]).item(), torch.mean(rgb_depths_tmp[valid_mask_tmp]).item(), torch.mean(tdepthmap_in_rgb[valid_mask_tmp]).item())
            #
            # # Second way to calculate the depth discrepancy
            # tpix_in_rgb = tpix_in_rgb.detach()
            # tdepth_in_rgb = tdepth_in_rgb.permute(0, 3, 1, 2).contiguous()  # (B, H, W, 1) -> (B, 1, H, W)
            # tpix_in_rgb = torch.stack([2 * tpix_in_rgb[:, :, :, 0] / (pred_w - 1) - 1, 2 * tpix_in_rgb[:, :, :, 1] / (pred_h - 1) - 1], dim=-1)
            # rgb_depth_c2ther = grid_sample(rgb_depths_tmp, tpix_in_rgb, mode='nearest', padding_mode='zeros', align_corners=True) # mode='bilinear'
            # valid_mask_tmp = (rgb_depth_c2ther > 0.1) & (tdepth_in_rgb > 0.1) & (rgb_depth_c2ther < self.config.max_depth) & (tdepth_in_rgb < self.config.max_depth)
            # diff = (rgb_depth_c2ther - tdepth_in_rgb).abs()
            # print("------------------- for debug, checking the warp, valid_mask_tmp and error-mean, error-median: ",
            #       torch.sum(valid_mask_tmp).item(), torch.mean(diff[valid_mask_tmp]).item(),
            #       torch.median(diff[valid_mask_tmp]).item(), torch.mean(rgb_depth_c2ther[valid_mask_tmp]).item(),
            #       torch.mean(tdepth_in_rgb[valid_mask_tmp]).item())
            #
            # ## For debug, vislize the input and supervision
            # vis_depth = colored_depthmap(tdepthmap_in_rgb[0].squeeze().cpu().numpy(), self.config.min_depth, self.config.max_depth).astype(np.uint8)
            # vis_depth1 = colored_depthmap(rgb_depths_tmp[0].squeeze().cpu().numpy(), self.config.min_depth, self.config.max_depth).astype(np.uint8)
            # vis_img = F.interpolate(images_rgb, size=(pred_h, pred_w), mode='nearest')
            # vis_img = 255.0 * vis_img[0].cpu().numpy()
            # vis_img = vis_img.transpose(1, 2, 0).astype(np.uint8)
            #
            # depth_error = (rgb_depths_tmp[0] - tdepthmap_in_rgb[0]).abs()
            # depth_error[valid_mask_tmp[0] == 0] = 0
            # vis_depth_error = colored_depthmap(depth_error.squeeze().cpu().numpy(), 0, 80).astype(np.uint8)
            #
            # vis_check_warp = np.hstack((vis_img, vis_depth, vis_depth1, vis_depth_error))
            # # cv2.imshow("rgb-warpedtdepth-rgbdepth-vis_depth_error", vis_check_warp)
            # # cv2.waitKey()
            # plt.imshow(vis_check_warp)
            # plt.axis('off')  # Hide the axes
            # plt.show()

            # ### Warp the rgb depth to thermal depth
            # pred_depths_rgb_tmp = F.interpolate(depths_gt_rgb, size=(pred_h, pred_w), mode='nearest').permute(0, 2, 3,
            #                                                                                           1).contiguous()  # ERROR: for debug only
            # thermal_depths_tmp = F.interpolate(depths_gt, size=(pred_h, pred_w),
            #                                mode='nearest')  # ERROR: for debug only
            # rgbpix_in_ther, rgbdepth_in_ther = remap_pixel_coordinates(pred_depths_rgb_tmp, rgb_intrinsics, thermal_intrinsics,
            #                                                      pose_cam1tocam2=None)  # pose_cam1tocam2
            # rgb_depthmap_in_ther = generate_depth_map_with_occlusion(rgbpix_in_ther, rgbdepth_in_ther, pred_h, pred_w)  # (B, 1, H, W)
            # valid_mask_tmp = (rgb_depthmap_in_ther > 0.1) & (thermal_depths_tmp > 0.1) & (
            #             rgb_depthmap_in_ther < self.config.max_depth) & (thermal_depths_tmp < self.config.max_depth)
            # print("---------------------- for debug, checking the warp, valid_mask_tmp and error-mean, error-median: ",
            #       torch.sum(valid_mask_tmp).item(),
            #       torch.mean((rgb_depthmap_in_ther - thermal_depths_tmp).abs()[valid_mask_tmp]).item(),
            #       torch.median((rgb_depthmap_in_ther - thermal_depths_tmp).abs()[valid_mask_tmp]).item(),
            #       torch.mean(thermal_depths_tmp[valid_mask_tmp]).item(),
            #       torch.mean(rgb_depthmap_in_ther[valid_mask_tmp]).item())

            # # For debug, vislize the input and supervision
            # vis_depth = colored_depthmap(rgb_depthmap_in_ther[0].squeeze().cpu().numpy(), self.config.min_depth, self.config.max_depth).astype(np.uint8)
            # vis_depth1 = colored_depthmap(thermal_depths_tmp[0].squeeze().cpu().numpy(), self.config.min_depth, self.config.max_depth).astype(np.uint8)
            # vis_img = F.interpolate(images_rgb, size=(pred_h, pred_w), mode='nearest')
            # vis_img = 255.0 * vis_img[0].cpu().numpy()
            # vis_img = vis_img.transpose(1, 2, 0).astype(np.uint8)
            #
            # depth_error = (rgb_depthmap_in_ther[0] - thermal_depths_tmp[0]).abs()
            # depth_error[valid_mask_tmp[0] == 0] = 0
            # vis_depth_error = colored_depthmap(depth_error.squeeze().cpu().numpy(), 0, 80).astype(np.uint8)
            #
            # vis_check_warp = np.hstack((vis_img, vis_depth, vis_depth1, vis_depth_error))
            # cv2.imshow("rgb-warpedtdepth-rgbdepth-vis_depth_error", vis_check_warp)
            # cv2.waitKey()



            if self.config.w_rgbt_ddiscrepancy > 0 or self.config.w_rgbt_kld > 0:
                ### Warp the thermal depth to rgb depth
                thermal_intrinsics = batch['intrinsics'].to(self.device) if 'intrinsics' in batch else batch['proj_matrix'].to(self.device)
                rgb_intrinsics = batch['rgb_intrinsics'].to(self.device) if 'rgb_intrinsics' in batch else batch['rgb_proj_matrix'].to(self.device)

                assert pred_depths_rgb.shape == pred_depths.shape

                # pose_t2rgb = batch['pose_t2rgb'].to(self.device)
                # pose_rgb2t = pose_t2rgb.inverse()
                # # Refer to https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6
                # # We should apply identity matrix, since P1 and P2 projection matrces, and both project points from coordiated of rectified cam1.
                # pose_t2rgb = torch.eye(4).unsqueeze(0).repeat(b, 1, 1).to(self.device)
                # pose_rgb2t = torch.eye(4).unsqueeze(0).repeat(b, 1, 1).to(self.device)

                pred_h, pred_w = pred_depths.shape[-2], pred_depths.shape[-1]
                scale_h, scale_w = pred_h/h, pred_w/w
                thermal_intrinsics[:, 0, :] = thermal_intrinsics[:, 0, :] * scale_w
                thermal_intrinsics[:, 1, :] = thermal_intrinsics[:, 1, :] * scale_h
                rgb_intrinsics[:, 0, :] = rgb_intrinsics[:, 0, :] * scale_w
                rgb_intrinsics[:, 1, :] = rgb_intrinsics[:, 1, :] * scale_h

                # print("------------- For debug, the shape of images, depths_gt, images_rgb, depths_gt_rgb, pred_depths, pred_depths_rgb: ",
                #       images.shape, depths_gt.shape, images_rgb.shape, depths_gt_rgb.shape, pred_depths.shape, pred_depths_rgb.shape)

                pred_depths_clone = pred_depths.permute(0,2,3,1).contiguous().clone() # (B, C, H, W) -> (B, H, W, C)
                tpix_in_rgb, tdepth_in_rgb = remap_pixel_coordinates(pred_depths_clone, thermal_intrinsics, rgb_intrinsics, pose_cam1tocam2=None) #pose_t2rgb
                tpix_in_rgb = tpix_in_rgb.detach()
                tdepth_in_rgb = tdepth_in_rgb.permute(0, 3, 1, 2).contiguous() # (B, H, W, 1) -> (B, 1, H, W)
                tpix_in_rgb = torch.stack([2 * tpix_in_rgb[:, :, :, 0] / (pred_w - 1) - 1, 2 * tpix_in_rgb[:, :, :, 1] / (pred_h - 1) - 1], dim=-1)
                rgb_depth_c2ther = grid_sample(pred_depths_rgb, tpix_in_rgb, mode='bilinear', padding_mode='zeros', align_corners=True)
                # pred_tdepthmap_in_rgb = generate_depth_map_with_occlusion(tpix_in_rgb, tdepth_in_rgb, pred_h, pred_w)  # (B, 1, H, W)


                pred_depths_rgb_clone = pred_depths_rgb.permute(0, 2, 3, 1).contiguous().clone()  # (B, C, H, W) -> (B, H, W, C)
                rgbpix_in_t, rgbdepth_in_t = remap_pixel_coordinates(pred_depths_rgb_clone, rgb_intrinsics, thermal_intrinsics, pose_cam1tocam2=None) #pose_rgb2t
                rgbpix_in_t = rgbpix_in_t.detach()
                rgbdepth_in_t = rgbdepth_in_t.permute(0, 3, 1, 2).contiguous() # (B, H, W, 1) -> (B, 1, H, W)
                rgbpix_in_t  = torch.stack([2 * rgbpix_in_t[:, :, :, 0] / (pred_w - 1) - 1, 2 * rgbpix_in_t[:, :, :, 1] / (pred_h - 1) - 1], dim=-1)
                ther_depth_c2rgb = grid_sample(pred_depths, rgbpix_in_t, mode='bilinear', padding_mode='zeros', align_corners=True)
                # pred_rgbdepthmap_in_t = generate_depth_map_with_occlusion(rgbpix_in_t, rgbdepth_in_t, pred_h, pred_w)  # (B, 1, H, W)


                # ### For debug, vislize the input and supervision
                # vis_depth_tinrgb = colored_depthmap(pred_tdepthmap_in_rgb[0].detach().squeeze().cpu().numpy(), 0, 7).astype(np.uint8)
                # vis_depth_rgb = colored_depthmap(pred_depths_rgb[0].detach().squeeze().cpu().numpy(), 0,
                #                                     7).astype(np.uint8)
                # vis_depth_rgbint = colored_depthmap(pred_rgbdepthmap_in_t[0].detach().squeeze().cpu().numpy(), 0,
                #                                     7).astype(np.uint8)
                # vis_depth_t = colored_depthmap(pred_depths[0].detach().squeeze().cpu().numpy(), 0,
                #                                     7).astype(np.uint8)
                # half_w = pred_w // 2
                # vis_check_warp = np.hstack((vis_depth_tinrgb, vis_depth_rgb, vis_depth_tinrgb[:, :half_w, :], vis_depth_rgb[:, half_w:, :], vis_depth_rgbint, vis_depth_t, vis_depth_rgbint[:, :half_w, :], vis_depth_t[:, half_w:, :]))
                # cv2.imshow("tinrgb-rgb-rgbint-t", vis_check_warp)
                # cv2.waitKey()

            # Depth discrepancy loss between rgb and thermal depth
            if self.config.w_rgbt_ddiscrepancy > 0:
                consistency_mask_warpther = (rgb_depth_c2ther > self.config.min_depth) & (rgb_depth_c2ther < self.config.max_depth) & (tdepth_in_rgb > self.config.min_depth) & (tdepth_in_rgb < self.config.max_depth)
                consistency_mask_warprgb = (ther_depth_c2rgb > self.config.min_depth) & (ther_depth_c2rgb < self.config.max_depth) & (rgbdepth_in_t > self.config.min_depth) & (rgbdepth_in_t < self.config.max_depth)

                discrepancy_loss_warpther = self.l1_loss_window(rgb_depth_c2ther.detach(), tdepth_in_rgb, mask=consistency_mask_warpther, interpolate=False, return_interpolated=False, window_size=self.config.windowloss_size, brelative_mask=True, error_threshold=0.08*self.config.max_depth)
                discrepancy_loss_warprgb = self.l1_loss_window(ther_depth_c2rgb, rgbdepth_in_t.detach(), mask=consistency_mask_warprgb, interpolate=False, return_interpolated=False, window_size=self.config.windowloss_size, brelative_mask=True, error_threshold=0.08*self.config.max_depth)
                loss = loss + self.config.w_rgbt_ddiscrepancy* 0.5 * (discrepancy_loss_warpther + discrepancy_loss_warprgb)
                losses[self.l1_loss_window.name] = 0.5 * (discrepancy_loss_warpther + discrepancy_loss_warprgb)

            # # KL divergence loss between rgb and thermal logits. Note that kld loss only works for pixel loss, while fail to work on window loss.
            # This is currently wrong, since it assumes rgb and thermal are co-registered (aligned).
            # if self.config.w_rgbt_kld > 0:
            #     consistency_mask = None
            #     if mask is not None:
            #         consistency_mask = mask & (pred_depths_rgb_intp > self.config.min_depth) & (pred_depths_rgb_intp < self.config.max_depth)
            #         consistency_mask = nn.functional.interpolate(consistency_mask.float(), size=(pred_h, pred_w), mode='nearest').to(torch.bool)
            #     else:
            #         consistency_mask = (pred_depths_rgb_intp > self.config.min_depth) & (pred_depths_rgb_intp < self.config.max_depth)
            #     pred_probs = output['probs']
            #     pred_probs_rgb = output_extra['probs']
            #     consistency_mask = consistency_mask.expand_as(pred_probs)
            #     # print("============= pred_probs, pred_probs_rgb ==============")
            #     # print("pred_probs.shape, pred_probs_rgb.shape: ", pred_probs.shape, pred_probs_rgb.shape)
            #     # print("pred_probs, sum, mean, std: ", pred_probs.sum(dim=1), pred_probs.mean(dim=1), pred_probs.std(dim=1))
            #     # print("pred_probs_rgb, sum, mean, std: ", pred_probs_rgb.sum(dim=1), pred_probs_rgb.mean(dim=1), pred_probs.std(dim=1))
            #     kld_loss = nn.functional.kl_div(pred_probs[consistency_mask], pred_probs_rgb[consistency_mask]) # pred_probs_rgb.detach()
            #     loss = loss + self.config.w_rgbt_kld * kld_loss
            #     losses['kld_loss'] = kld_loss

            # Smooth loss
            if self.config.w_depth_smooth > 0:
                # print("------------------- pred_depths_rgb_intp.shape, images.shape: ", pred_depths_rgb_intp.shape, images.shape)
                #if pred_depths_intp is None:
                #    pred_depths_intp = nn.functional.interpolate(pred_depths, images.shape[-2:], mode='bilinear', align_corners=True)
                #smooth_loss = get_disparity_smooth_loss(pred_depths_intp, images, bexlude_top20=True)
                
                smooth_loss = 0.0
                if pred_depths_rgb_intp is None:
                    pred_depths_rgb_intp = nn.functional.interpolate(pred_depths_rgb, images_rgb.shape[-2:], mode='bilinear', align_corners=True)
                smooth_loss = smooth_loss + get_disparity_smooth_loss(pred_depths_rgb_intp, images_rgb, bexlude_top20=True)

                loss = loss + self.config.w_depth_smooth * smooth_loss
                losses['depth_smooth_loss'] = smooth_loss

            # NLL loss
            if self.config.w_rgbt_nll > 0:
                consistency_l1loss = output_extra['consistency_l1loss']
                if consistency_l1loss is not None:
                    loss = (loss + self.config.w_rgbt_nll_l1 * consistency_l1loss) # ERROR
                else:
                    print("Warning, the nll_l1_loss is None!")
                losses['nll_l1_loss'] = consistency_l1loss

                conf_depth_rgb = output_extra['conf']  # var_inv, [0~1]

                if self.config.w_si > 0 or self.config.w_si_thermal > 0: # In this case, we will use the depths_gt_rgb as supervision signal.
                    mask_nll =  (depths_gt_rgb > self.config.min_depth) & (depths_gt_rgb < self.config.max_depth)
                    nll_loss = self.nll_loss(pred_depths_rgb_intp.detach(), depths_gt_rgb, conf_depth_rgb, mask_nll,  ignore_bigdiff=True)
                    if nll_loss is not None:
                        loss = (loss + self.config.w_rgbt_nll * nll_loss)
                    else:
                        print("Warning, the nll_loss is None!")
                    losses['nll_loss'] = nll_loss

                # Smooth loss for the predicted conf
                if self.config.w_rgbt_nll_l1_smooth > 0:
                    pred_depths_rgb_rel = output_extra['rel_depth'].unsqueeze(1) # (B, 1, H, W)
                    nll_l1_smooth_loss = get_disparity_smooth_loss(conf_depth_rgb, pred_depths_rgb_rel.detach(), bexlude_top20=True)
                    loss = (loss + self.config.w_rgbt_nll_l1_smooth * nll_l1_smooth_loss)
                    losses['nll_l1_smooth_loss'] = nll_l1_smooth_loss

            if self.config.w_grad > 0:
                l_grad = self.grad_loss(pred_depths_rgb_intp, depths_gt_rgb, mask=mask)
                loss = loss + self.config.w_grad * l_grad
                losses[self.grad_loss.name] = l_grad


        self.scaler.scale(loss).backward()

        if self.config.clip_grad > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad)

        self.scaler.step(self.optimizer)

        if self.should_log and (self.step % int(self.config.log_images_every * self.iters_per_epoch)) == 0:
            # -99 is treated as invalid depth in the log_images function and is colored grey.
            depths_gt[torch.logical_not(mask)] = -99
            depths_gt_rgb[torch.logical_not(mask_rgb)] = -99
            if self.config.w_rgbt_nll > 0:
                rgb_nll_ls_mask = output_extra['consistency_mask']
                rgb_nll_ls_mask =(rgb_nll_ls_mask[0, ...]).float() # (1, H, W)
                self.log_images(rgb={"Input_RGB": images_rgb[0, ...], "Input_Ther": images[0, ...]}, depth={"GT_RGB": depths_gt_rgb[0], "PredictedRGB": pred_depths_rgb[0], "GT": depths_gt[0], "PredictedTher": pred_depths[0]},
                                scalar_field={"RGB_conf": conf_depth_rgb[0, ...], "Consistency_mask": rgb_nll_ls_mask}, prefix="Train", min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])
            else:
                self.log_images(rgb={"Input_RGB": images_rgb[0, ...], "Input_Ther": images[0, ...]}, depth={"GT_RGB": depths_gt_rgb[0], "PredictedRGB": pred_depths_rgb[0], "GT": depths_gt[0], "PredictedTher": pred_depths[0]},
                                prefix="Train", min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])

            if self.config.get("log_rel", False):
                self.log_images(
                    scalar_field={"RelPred": output["relative_depth"][0]}, prefix="TrainRel")

        self.scaler.update()
        self.optimizer.zero_grad()

        return losses
    
    @torch.no_grad()
    def eval_infer(self, x, y):
        with amp.autocast(enabled=self.config.use_amp):
            m = self.model.module if self.config.multigpu else self.model
            out, out_extra = m(x, y)
            pred_depths = out['metric_depth']
            pred_depths_extra = out_extra['metric_depth']
        return pred_depths, pred_depths_extra

    @torch.no_grad()
    def crop_aware_infer(self, x):
        # if we are not avoiding the black border, we can just use the normal inference
        if not self.config.get("avoid_boundary", False):
            return self.eval_infer(x)
        
        # otherwise, we need to crop the image to avoid the black border
        # For now, this may be a bit slow due to converting to numpy and back
        # We assume no normalization is done on the input image

        # get the black border
        assert x.shape[0] == 1, "Only batch size 1 is supported for now"
        x_pil = transforms.ToPILImage()(x[0].cpu())
        x_np = np.array(x_pil, dtype=np.uint8)
        black_border_params = get_black_border(x_np)
        top, bottom, left, right = black_border_params.top, black_border_params.bottom, black_border_params.left, black_border_params.right
        x_np_cropped = x_np[top:bottom, left:right, :]
        x_cropped = transforms.ToTensor()(Image.fromarray(x_np_cropped))

        # run inference on the cropped image
        pred_depths_cropped = self.eval_infer(x_cropped.unsqueeze(0).to(self.device))

        # resize the prediction to x_np_cropped's size
        pred_depths_cropped = nn.functional.interpolate(
            pred_depths_cropped, size=(x_np_cropped.shape[0], x_np_cropped.shape[1]), mode="bilinear", align_corners=False)
        

        # pad the prediction back to the original size
        pred_depths = torch.zeros((1, 1, x_np.shape[0], x_np.shape[1]), device=pred_depths_cropped.device, dtype=pred_depths_cropped.dtype)
        pred_depths[:, :, top:bottom, left:right] = pred_depths_cropped

        return pred_depths

    def validate_on_batch(self, batch, val_step):
        images = batch['image'].to(self.device) # Thermal images
        depths_gt = batch['depth'].to(self.device)
        dataset = batch['dataset'][0]
        mask = batch["mask"].to(self.device)
        if 'has_valid_depth' in batch:
            if not batch['has_valid_depth'][0]:
                return None, None


        depths_gt = depths_gt.squeeze().unsqueeze(0).unsqueeze(0)
        mask = mask.squeeze().unsqueeze(0).unsqueeze(0)

        b, c, h, w = images.size()
        images_rgb = batch['rgb_image'].to(self.device)  # rgb image
        depths_gt_rgb = batch['rgb_depth'].to(self.device)  # rgb depth
        mask_rgb = batch["rgb_mask"].to(self.device).to(torch.bool)
        depths_gt_rgb = depths_gt_rgb.squeeze().unsqueeze(0).unsqueeze(0)
        mask_rgb = mask_rgb.squeeze().unsqueeze(0).unsqueeze(0)

        if dataset == 'nyu':
            # pred_depths = self.crop_aware_infer(images)
            raise NotImplementedError("NYU dataset is not supported for rgbt_trainer......")
        else:
            pred_depths, pred_depths_extra = self.eval_infer(images, images_rgb)
        pred_depths = pred_depths.squeeze().unsqueeze(0).unsqueeze(0) # Thermal depth
        pred_depths_extra = pred_depths_extra.squeeze().unsqueeze(0).unsqueeze(0) # RGB depth

        with amp.autocast(enabled=self.config.use_amp):
            l_depth = self.silog_loss(
                pred_depths, depths_gt, mask=mask.to(torch.bool), interpolate=True)

        metrics = compute_metrics(depths_gt, pred_depths, **self.config)
        losses = {f"{self.silog_loss.name}": l_depth.item()}

        if val_step == 1 and self.should_log:
            depths_gt[torch.logical_not(mask)] = -99
            depths_gt_rgb[torch.logical_not(mask_rgb)] = -99
            self.log_images(rgb={"Input_RGB": images_rgb[0, ...], "Input_Ther": images[0, ...]}, depth={"GT_RGB": depths_gt_rgb[0], "PredictedRGB": pred_depths_extra[0], "GT": depths_gt[0], "PredictedTher": pred_depths[0]}, prefix="Test",
                            min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])

        return metrics, losses
