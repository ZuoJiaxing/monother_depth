import os

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.utils.data.distributed

import sys
import os
sys.path.append(os.getcwd())

import glob
import tqdm 
import skimage 
import random

import matplotlib.pyplot as plt
from PIL import Image
import imageio
from imageio import imread
from torch.nn.functional import grid_sample
import torchmetrics
import torch.nn.functional as F
# from torchmetrics.functional import structural_similarity_index_measure as ssim

from zoedepth.utils.config import check_choices
from zoedepth.utils.depth_utils import remap_pixel_coordinates




class MS2DataLoader:
    def __init__(self, config, mode, modality='rgbtd', **kwargs):
        """
        Data loader for depth datasets

        Args:
            config (dict): Config dictionary. Refer to utils/config.py
            mode (str): "train" or "val', 'test_day', 'test_night', 'test_rainy"
        """

        self.config = config

        check_choices('Modality', modality, ['rgbtd', 'rgbd', 'td'])
        print("$$$$$$$$$$$$$$$$$$$$$ Modality: ", modality)

        geometric_transforms = A.Compose([
            A.SmallestMaxSize(max_size=min(self.config.input_height, self.config.input_width), p=1),
            A.RandomCrop(self.config.input_height, self.config.input_width),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=1.0, p=0.5), # between -1 and 1 degrees
        ], additional_targets={
            'image': 'image',
            'depth': 'mask',
            'rgb_image': 'image',
            'rgb_depth': 'mask',
        })

        photometric_transforms = A.Compose([
            A.RandomGamma(gamma_limit=(90, 110), p=0.5), # between 0.9 and 1.1
            A.ColorJitter(hue=(-0.1, 0.1), p=1),
        ])



        if mode == 'train':
            self.training_samples = MS2Dataset(
                self.config.data_path, 
                self.config.split_path, 
                mode, 
                min_depth=self.config.min_depth,
                max_depth=self.config.max_depth,
                geometric_transforms=None, # geometric_transforms
                photometric_transforms=None, # photometric_transforms,
                modality=modality,
                auto_scale_thermal=self.config.auto_scale_thermal,
                rearr_thermal=self.config.rearr_thermal

            )

            self.train_sampler = None
            if config.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.training_samples)

            self.data = torch.utils.data.DataLoader(
                self.training_samples,
                batch_size=config.batch_size,
                shuffle= False, # (self.train_sampler is None),
                num_workers=config.workers,
                pin_memory=True,
                persistent_workers=True,
                sampler=self.train_sampler,
            )

        elif mode in ['online_eval', 'val', 'test_day', 'test_night', 'test_rainy']:
            if mode == 'online_eval':
                mode = 'val'

            self.testing_samples = self.training_samples = MS2Dataset(
                self.config.data_path, 
                self.config.split_path, 
                mode, 
                min_depth=self.config.min_depth,
                max_depth=self.config.max_depth,
                geometric_transforms=None, 
                photometric_transforms=None, 
                modality=modality
            )

            self.data = torch.utils.data.DataLoader(
                self.testing_samples, 
                1,
                shuffle=False,
                num_workers=config.workers,
                pin_memory=False,
                sampler=None,
            )

        else:
            raise ValueError("Invalid mode: {}, mode should be one of train, val, test_day, test_night, test_rainy".format(mode))





class MS2Dataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split_path, mode, min_depth=0.001, max_depth=80, 
                 geometric_transforms=None, photometric_transforms=None, modality='rgbtd', auto_scale_thermal=True,
                rearr_thermal=False, **kwargs):
        """
        Modalidty: list of modalities to load. Options: 'td', 'rgbd', 'rgbtd', 
        which denotes thermal+depth, rgb + depth, and rgb+thermal+thermaldepth_rgbdepth respectively
        """

        self.modality = modality
        self.mode = mode
        
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.auto_scale_thermal = auto_scale_thermal
        self.rearr_thermal = rearr_thermal
        self.do_normalize = False

        check_choices('Modality', modality, ['rgbtd', 'rgbd', 'td'])
        check_choices('Mode', mode, ['train', 'val', 'test_day', 'test_night', 'test_rainy'])

        
        folder_list_path = os.path.join(split_path, '{}_list.txt'.format(self.mode))
        assert os.path.exists(folder_list_path), "Folder list path does not exist: {}".format(folder_list_path)

        self.sequences = []
        with open(folder_list_path, 'r') as f:
            for fp in f.readlines():
                self.sequences.append(fp.strip())

        self.calib = {}
        self.filenames = []
        for seq in self.sequences:
            sync_path = os.path.join(data_root, 'sync_data', seq)

            thermal_folder_path = os.path.join(sync_path, 'thr', 'img_left')
            rgb_folder_path = os.path.join(sync_path, 'rgb', 'img_left')
            calib_path = os.path.join(sync_path, 'calib.npy')

            calib_dict = np.load(calib_path, allow_pickle=True).item()


            ext_NIR2THR = np.concatenate([calib_dict['R_nir2thr'], calib_dict['T_nir2thr'] * 0.001], axis=1)  # mm -> m scale conversion.
            ext_NIR2RGB = np.concatenate([calib_dict['R_nir2rgb'], calib_dict['T_nir2rgb'] * 0.001], axis=1)
            ext_THR2NIR = np.linalg.inv(np.concatenate([ext_NIR2THR, [[0, 0, 0, 1]]], axis=0))
            ext_THR2RGB = np.matmul(np.concatenate([ext_NIR2RGB, [[0, 0, 0, 1]]], axis=0), ext_THR2NIR)
            calib_dict['pose_t2rgb'] = ext_THR2RGB


            ext_RGB_L2R = np.concatenate([calib_dict['R_rgbR'], calib_dict['T_rgbR'] * 0.001], axis=1)
            calib_dict['pose_rgb_l2r'] = ext_RGB_L2R

            ext_Ther_L2R = np.concatenate([calib_dict['R_thrR'], calib_dict['T_thrR'] * 0.001], axis=1)
            calib_dict['pose_ther_l2r'] = ext_Ther_L2R


            self.calib[seq] = calib_dict

            # print("calib: ", self.calib[seq]['K_nirL'])
            # print("calib: ", self.calib[seq]['K_rgbL'])
            # print("calib: ", self.calib[seq]['K_thrL'])
            #
            # print("calib: ", self.calib[seq]['R_nir2rgb'])
            # print("calib: ", self.calib[seq]['T_nir2rgb'])
            # print("calib: ", self.calib[seq]['R_nir2thr'])
            # print("calib: ", self.calib[seq]['T_nir2thr'])
            #
            # print("ext_THR2RGB:", ext_THR2RGB)

            proj_depth_path = os.path.join(data_root, 'proj_depth', seq)
            thermal_depth_folder_path = os.path.join(proj_depth_path, 'thr', 'depth_filtered')
            rgb_depth_folder_path = os.path.join(proj_depth_path, 'rgb', 'depth_filtered')

            thermal_files = sorted(glob.glob(os.path.join(thermal_folder_path, '*.png')))
            for thermal_file in tqdm.tqdm(thermal_files):
                filename = os.path.basename(thermal_file)
                thermal_depth_file = os.path.join(thermal_depth_folder_path, filename)
                rgb_file = os.path.join(rgb_folder_path, filename)
                rgb_depth_file = os.path.join(rgb_depth_folder_path, filename)

                assert os.path.exists(thermal_file), "Thermal file does not exist: {}".format(thermal_file)
                assert os.path.exists(thermal_depth_file), "Thermal depth file does not exist: {}".format(thermal_depth_file)
                assert os.path.exists(rgb_file), "RGB file does not exist: {}".format(rgb_file)
                assert os.path.exists(rgb_depth_file), "RGB depth file does not exist: {}".format(rgb_depth_file)

                # ### For debug only
                # if len(self.filenames)>100:
                #     break

                self.filenames.append({
                    'seq': seq,
                    'image_path': thermal_file, 
                    'depth_path': thermal_depth_file, 
                    'rgb_path': rgb_file, 
                    'rgb_depth_path': rgb_depth_file, 
                    'calibration': calib_dict,
                })

        self.geometric_transforms = geometric_transforms
        self.photometric_transforms = photometric_transforms

        self.thermal_normalize = A.Compose([
            A.Normalize(mean=[0.45], std=[0.225], max_pixel_value=1.0),
        ])

        self.rgb_normalize = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
        ])

        
    def rawthermal_autoScale(self, img, bits=16, minmax_norm=False, clahe=False): # img [0, 1] or [0, 255]
        img = img / (2**bits - 1) # [0, 1]

        if minmax_norm:
            threshold_lo = np.percentile(img, 2)
            threshold_hi = np.percentile(img, 98)
            img = np.clip((img - threshold_lo) / (threshold_hi - threshold_lo), 0, 1)
        
        if clahe:
            # Contrast Limited Adaptive Histogram Equalization
            img = skimage.exposure.equalize_adapthist(img, clip_limit=0.015)

        return img.astype(np.float32)

    def load_rgb_as_float(self, path):
        image = Image.open(path)
        image = np.asarray(image, dtype=np.float32) / 255.0
        return image  # HW3

    def load_thermal_as_float(self, path):
        return imageio.v2.imread(path).astype(np.float32)  # HW

    def load_as_float_img(self, path):
        img = imread(path).astype(np.float32)
        if len(img.shape) == 2:  # for NIR and thermal images
            img = np.expand_dims(img, axis=2)
        return img
    def load_as_float_depth(self, path):
        if 'png' in path:
            depth = np.array(imread(path).astype(np.float32))
        elif 'npy' in path:
            depth = np.load(path).astype(np.float32)
        elif 'mat' in path:
            depth = self.loadmat(path).astype(np.float32)
        return depth

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]

        # Get sample paths
        image_path = sample_path['image_path']
        depth_path = sample_path['depth_path']
        rgb_path = sample_path['rgb_path']
        rgb_depth_path = sample_path['rgb_depth_path']
        calib = self.calib[sample_path['seq']]

        thermal_proj_matrix = calib['K_thrL'].astype(np.float32)
        rgb_proj_matrix = calib['K_rgbL'].astype(np.float32)
        pose_t2rgb = calib['pose_t2rgb'].astype(np.float32)

        pose_ther_l2r = calib['pose_ther_l2r'].astype(np.float32)
        pose_rgb_l2r = calib['pose_rgb_l2r'].astype(np.float32)

        # Read data
        # thermal_img_raw = cv2.imread(image_path, -1)
        # rgb_img = cv2.imread(rgb_path, 1).astype(np.float32)
        # rgb_img = rgb_img/255.0  # (h, w, 3)
        # # rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) / 255.0 # (h, w, 3)  # This is wrong

        thermal_img_raw = self.load_thermal_as_float(image_path)
        thermal_right_path = image_path.replace('img_left', 'img_right')
        thermal_img_right_raw = self.load_thermal_as_float(thermal_right_path)

        rgb_img = self.load_rgb_as_float(rgb_path)
        image_right_path = rgb_path.replace('img_left', 'img_right')
        rgb_img_right = self.load_rgb_as_float(image_right_path)
        
        # MS2 stores depth in metric scale depth * 256
        thermal_depth_gt = cv2.imread(depth_path, -1).astype(np.float32) / 256.0
        rgb_depth_gt = cv2.imread(rgb_depth_path, -1).astype(np.float32) / 256.0

        thermal_h, thermal_w = thermal_img_raw.shape
        rgb_h, rgb_w = rgb_img.shape[0], rgb_img.shape[1]

        # ########################################################################################################################
        # ### Resize the RGB images to the thermal image size
        # if self.modality=="rgbtd" and (thermal_h != rgb_h or thermal_w != rgb_w):
        #     rgb_img = cv2.resize(rgb_img, (thermal_w, thermal_h), interpolation=cv2.INTER_LINEAR) # (1224, 384) -> (640, 256)  # cv2.INTER_NEAREST
        #     rgb_depth_gt = cv2.resize(rgb_depth_gt, (thermal_w, thermal_h), interpolation=cv2.INTER_NEAREST) # (1224, 384) -> (640, 256) # cv2.INTER_NEAREST
        #     rgb_proj_matrix[0, :] = rgb_proj_matrix[0, :] * thermal_w/rgb_w
        #     rgb_proj_matrix[1, :] = rgb_proj_matrix[1, :] * thermal_h/rgb_h
        #
        #     rgb_img_right = cv2.resize(rgb_img_right, (thermal_w, thermal_h), interpolation=cv2.INTER_LINEAR) # (1224, 384) -> (640, 256)  # cv2.INTER_NEAREST


        # Normalize the thermal image
        # print("--------------------------- For debug, self.auto_scale_thermal: ", self.auto_scale_thermal)
        thermal_img = self.rawthermal_autoScale(thermal_img_raw, bits=16, minmax_norm=self.auto_scale_thermal, clahe=self.rearr_thermal)
        thermal_img_right = self.rawthermal_autoScale(thermal_img_right_raw, bits=16, minmax_norm=self.auto_scale_thermal, clahe=self.rearr_thermal)

        # Transform images only
        if self.photometric_transforms is not None:
            thermal_img = self.photometric_transforms(image=thermal_img)['image']
            thermal_img_right = self.photometric_transforms(image=thermal_img_right)['image']
            rgb_img = self.photometric_transforms(image=rgb_img)['image']
            rgb_img_right = self.photometric_transforms(image=rgb_img_right)['image']


        # # Transform all data (images and depths) geometrically
        # if self.geometric_transforms is not None:
        #     transformed_data = self.geometric_transforms(
        #         image=thermal_img,
        #         depth=thermal_depth_gt,
        #         rgb_image=rgb_img,
        #         rgb_depth=rgb_depth_gt
        #     )
        #
        #     thermal_img = transformed_data['image']
        #     thermal_depth_gt = transformed_data['depth']
        #     rgb_img = transformed_data['rgb_image']
        #     rgb_depth_gt = transformed_data['rgb_depth']

        
        # # Normalize thermal and rgb images to zero mean, unit std
        if self.do_normalize:
            thermal_img = self.thermal_normalize(image=thermal_img)['image']
            thermal_img_right = self.thermal_normalize(image=thermal_img_right)['image']
            rgb_img = self.rgb_normalize(image=rgb_img)['image']
            rgb_img_right = self.rgb_normalize(image=rgb_img_right)['image']

        # Get valid masks for depth
        thermal_mask = np.logical_and(thermal_depth_gt > self.min_depth,
                            thermal_depth_gt < self.max_depth).squeeze()
        rgb_mask = np.logical_and(rgb_depth_gt > self.min_depth,
                            rgb_depth_gt < self.max_depth).squeeze()

        # Convert to tensor
        thermal_img = ToTensorV2()(image=thermal_img)['image']
        thermal_img_right = ToTensorV2()(image=thermal_img_right)['image']
        rgb_img = ToTensorV2()(image=rgb_img)['image']
        rgb_img_right = ToTensorV2()(image=rgb_img_right)['image']
        thermal_depth_gt = ToTensorV2()(image=thermal_depth_gt)['image']
        rgb_depth_gt = ToTensorV2()(image=rgb_depth_gt)['image']
        thermal_mask = ToTensorV2()(image=thermal_mask)['image']
        rgb_mask = ToTensorV2()(image=rgb_mask)['image']

        # Duplicate the channeles of thermal image if it is single channel
        if thermal_img.shape[0] == 1:
            thermal_img = torch.cat([thermal_img]*3, dim=0)

        if thermal_img_right.shape[0] == 1:
            thermal_img_right = torch.cat([thermal_img_right]*3, dim=0)

        if self.modality == 'rgbtd':
            sample = {
                'dataset': 'ms2thermal',
                'image': thermal_img,
                'image_right': thermal_img_right,
                'depth': thermal_depth_gt,
                'pose_ther_l2r': pose_ther_l2r,

                'rgb_image': rgb_img,
                'rgb_image_right': rgb_img_right,
                'rgb_depth': rgb_depth_gt,
                'pose_rgb_l2r': pose_rgb_l2r,

                'has_valid_depth': True, # MS2 always has depth
                'intrinsics': thermal_proj_matrix,
                'rgb_intrinsics': rgb_proj_matrix,
                'pose_t2rgb': pose_t2rgb,
                # 'focal' : 0, # dummy value

                'image_path': image_path,
                'depth_path': depth_path,
                'rgb_depth_path': rgb_depth_path,

                'mask': thermal_mask,
                'rgb_mask':rgb_mask
            }
        elif self.modality == 'rgbd':
            sample = {
                'dataset': 'ms2thermal',
                'image': rgb_img,
                'depth': rgb_depth_gt,
                'intrinsics': rgb_proj_matrix,
                'mask': rgb_mask,
                'has_valid_depth': True, # MS2 always has depth
                'image_path': rgb_path,
                'depth_path': rgb_depth_path
            }
        elif self.modality == 'td':
            sample = {
                'dataset': 'ms2thermal',
                'image': thermal_img,
                'depth': thermal_depth_gt,
                'intrinsics': thermal_proj_matrix,
                'mask': thermal_mask,
                'has_valid_depth': True, # MS2 always has depth
                'image_path': image_path,
                'depth_path': depth_path
            }

        return sample


def get_ms2thermal_loader(config, mode, modality='rgbtd', **kwargs):
    dataloader = MS2DataLoader(config=config, mode=mode, modality=modality, **kwargs).data
    return dataloader

def colorize_depth_img(depth_img, min_depth, max_depth):
    vis_depth_img = (depth_img - min_depth) / (max_depth - min_depth)
    vis_depth_img = np.clip(vis_depth_img, 0, 1)
    vis_depth_img = 1.0 - vis_depth_img
    vis_depth_img = cv2.applyColorMap((vis_depth_img * 255).astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32) / 255.0
    return vis_depth_img # (h, w, 3)


def visualize_rgb_depth(rgb_image, depth_map, min_depth=0.001, max_depth=80):
    """
    Visualize an RGB image and its corresponding depth map in the same figure.

    Parameters:
    rgb_image: (H, W, 3) - RGB image in numpy array format
    depth_map: (H, W) - depth map in numpy array format
    """
    # Ensure depth_map has the same height and width as rgb_image
    assert rgb_image.shape[:2] == depth_map.shape, "Shape mismatch between RGB image and depth map"

    # Create a mask where depth is greater than zero
    depth_mask = depth_map > 0
    vis_depth_img = colorize_depth_img(depth_map, min_depth, max_depth)

    # Create an output image
    output_image = np.copy(rgb_image)

    # Replace RGB values with depth values where depth is greater than zero
    output_image[depth_mask] = vis_depth_img[depth_mask]
    return output_image


def plot_feat_matching(image1, image2, match_pairs):
    """
    image1: (3, H, W) - RGB image in tensor
    image2: (3, H, W) - RGB image in tensor
    match_pairs: (H, W, 2) - The pixel coordinates of the matching points in image2 corresponding to image1
    """
    H, W = image1.shape[-2:]
    assert image1.shape == image2.shape, "Image shapes do not match"
    assert match_pairs.shape == (H, W, 2), f"Shape mismatch in match_pairs, match_paris shape: {match_pairs.shape}"

    image1 = image1.permute(1, 2, 0).cpu().numpy() # (H, W, 3)
    image2 = image2.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

    # Create a figure and two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.tight_layout()

    # Plot the first image
    axs[0].imshow(image1)
    axs[0].set_title('Image 1')

    # Plot the second image
    axs[1].imshow(image2)
    axs[1].set_title('Image 2')

    # Plot correspondences as lines
    for i in range(0, H, 10):
        for j in range(0, W, 10):
            x1, y1 = j, i
            x2, y2 = match_pairs[i, j]
            if x2>0 and y2>0 and x2<W and y2<H:
                color = [random.random() for _ in range(3)]  # Generate a random color
                axs[0].plot(x1, y1, 'o', color=color)  # Plot point in image 1
                axs[1].plot(x2, y2, 'o', color=color)  # Plot point in image 2
                fig.add_artist(plt.Line2D((x1, W + x2), (y1, y2), color=color, linewidth=2))   # Create line connecting the points

    # Adjust the layout
    plt.show()

def patch_ssim(patch1, patch2, data_range=1.0, K1=0.01, K2=0.03):
    """
    Calculate the SSIM index between two image patches. The patches should be of shape (B, C, Patch_size, Patch_size).
    """
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # Means
    mu1 = patch1.mean(dim=(2, 3), keepdim=True)
    mu2 = patch2.mean(dim=(2, 3), keepdim=True)

    # Variances
    sigma1_sq = ((patch1 - mu1) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma2_sq = ((patch2 - mu2) ** 2).mean(dim=(2, 3), keepdim=True)

    # Covariance
    sigma12 = ((patch1 - mu1) * (patch2 - mu2)).mean(dim=(2, 3), keepdim=True)

    # SSIM calculation
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)) # (B, 3, 1, 1)
    return ssim_map.mean(dim=1).squeeze(dim=-1).squeeze(dim=-1) # (B)


def get_patches_with_gridsample(image, correspondence, correspondence_mask, patch_size=7):
    """
    image2: (B, C, H, W) - Image tensor from which patches are to be sampled
    correspondence: (B, H, W, 2) - Correspondence tensor containing the pixel coordinates of the matching points in image2

    """
    B, C, H, W = image.shape
    device = image.device

    # Prepare the grid for sampling patches from image2
    x_coords, y_coords = correspondence[..., 0], correspondence[..., 1] # (B, H, W)

    # Create grids for each patch element
    patch_range = torch.arange(-(patch_size//2), patch_size//2 + 1, device=device)
    grid_y, grid_x = torch.meshgrid(patch_range, patch_range, indexing='ij')
    grid_y = grid_y.contiguous().view(-1)
    grid_x = grid_x.contiguous().view(-1)

    # Create the full grid for all pixels
    y_coords_grid = y_coords.view(B, -1, 1) + grid_y.view(1, 1, -1)
    x_coords_grid = x_coords.view(B, -1, 1) + grid_x.view(1, 1, -1)

    # Normalize the grid coordinates
    y_coords_grid = (2.0 * y_coords_grid / (H - 1)) - 1
    x_coords_grid = (2.0 * x_coords_grid / (W - 1)) - 1

    # Stack and expand dimensions to match the grid format for grid_sample
    grid = torch.stack((x_coords_grid, y_coords_grid), dim=-1).view(B, H, W, patch_size, patch_size, 2)

    # Reshape the grid to (B, H*W*patch_size*patch_size, 2)
    grid = grid.view(B, H*W,  patch_size, patch_size, 2)
    grid = grid.view(B, H*W, patch_size*patch_size, 2)

    # Sample patches from image2 using grid_sample
    patches = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=True) # Shape: (B, C, H*W, patch_size*patch_size)

    # Reshape patches2 to (B, C, H, W, patch_size, patch_size)
    patches = patches.view(B, C, H, W, patch_size*patch_size)
    patches = patches.view(B, C, H, W, patch_size,patch_size)

    # Mask out invalid patches
    patches *= correspondence_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

    return patches.permute(0, 1, 4, 5, 2, 3).contiguous()  # (B, C, patch_size, patch_size, H, W)


def vectorized_ssim(patches1, patches2):
    """
    patches1 and patches2 are with shape [B, C, patch_size, patch_size, H, W]
    """
    B, C, patch_size, _, H, W = patches1.shape
    device = patches1.device

    # Flatten the patches dimensions for batch processing
    patches1_flat = patches1.permute(0, 4, 5, 1, 2, 3).reshape(-1, C, patch_size, patch_size)
    patches2_flat = patches2.permute(0, 4, 5, 1, 2, 3).reshape(-1, C, patch_size, patch_size)

    # Compute SSIM for each patch in a vectorized manner
    ssim_values_flat = patch_ssim(patches1_flat, patches2_flat)

    # Reshape back to original dimensions
    ssim_values_flat = ssim_values_flat.reshape(B, H, W)

    # Set NaN elements to zero
    ssim_values_flat[torch.isnan(ssim_values_flat)] = 0.0

    return ssim_values_flat


def calculate_ssim_loss_batched(image1, image2, correspondence, correspondence_mask, patch_size=7):
    """
    Calculate the SSIM loss for known feature correspondences in two images with batch input.

    Args:
    - image1: Tensor of shape (B, C, H, W) representing the first batch of images.
    - image2: Tensor of shape (B, C, H, W) representing the second batch of images.
    - correspondence: Tensor of shape (B, H, W, 2) representing feature correspondences.
    - correspondence_mask: (B, H, W) - correspondence_mask for the valid correspondences
    - patch_size: Size of the patch for SSIM calculation.

    Returns:
    - ssim_loss: The SSIM loss.
    """
    B, C, H, W = image1.shape
    device = image1.device

    # Extract patches around each correspondence
    patches1 = F.unfold(image1, kernel_size=patch_size, padding=patch_size // 2)
    # patches1 = patches1.view(B, C, patch_size, patch_size, H, W).permute(0, 4, 5, 1, 2, 3)
    patches1 = patches1.permute(0, 2, 1).view(B, H, W, patch_size, patch_size, C)
    patches1 = patches1.permute(0, 5, 3, 4, 1, 2) # (B, C, patch_size, patch_size, H, W)

    # # Way1: Get patches by for loop
    # patches2 = torch.zeros_like(patches1, device=device)
    # for b in range(B):
    #     for i in range(H):
    #         for j in range(W):
    #             if not correspondence_mask[b, i, j]:
    #                 continue
    #             x, y = correspondence[b, i, j]
    #             y, x = int(y.item()), int(x.item())
    #             y_start, y_end = max(0, y - patch_size // 2), min(H, y + patch_size // 2 + 1)
    #             x_start, x_end = max(0, x - patch_size // 2), min(W, x + patch_size // 2 + 1)
    #             assert y_start<y_end and x_start<x_end, f"x: {x}, y: {y}, y_start: {y_start}, y_end: {y_end}, x_start: {x_start}, x_end: {x_end}"
    #             patch = image2[b, :, y_start:y_end, x_start:x_end]
    #             # Ensure the patch has the correct size by padding if necessary
    #             patch_padded = torch.zeros(C, patch_size, patch_size, device=device)
    #             patch_padded[:, :patch.shape[1], :patch.shape[2]] = patch
    #             patches2[b, :, :, :, i, j] = patch_padded

    # Way2: Get patches by for grid_sample
    patches2 = get_patches_with_gridsample(image2, correspondence, correspondence_mask, patch_size=patch_size) # (B, C, patch_size, patch_size, H, W)

    # # Compute SSIM for each patch
    # Way 1: compute with foor loop
    # ssim_values = torch.zeros(B, H, W, device=device)
    # for i in range(H):
    #     for j in range(W):
    #         # if not correspondence_mask[:, i, j]:
    #         #     continue
    #         ssim_value = patch_ssim(patches1[:, :, :, :, i, j], patches2[:, :, :, :, i, j]) # (B)
    #         ssim_value[torch.isnan(ssim_value)] =0.0
    #         ssim_values[:, i, j] = ssim_value

    # Way 2: compute with vectorized_ssim
    ssim_values = vectorized_ssim(patches1, patches2) # (B, H, W)
    ssim_values = ssim_values*correspondence_mask

    # Compute the SSIM loss (1 - mean SSIM value)
    ssim_loss = 1 - ssim_values

    return ssim_loss # (B, H, W)

def get_lr_consistency_mask(depth_cam1, intrinsics_cam1, intrinsics_cam2, pose_cam1_to_cam2, rgb_img_raw, rgb_img_right_raw, l1_thresh=0.9, ssim_thresh=0.9):
    """
    Calculate the left-right consistency mask for a depth map of cam1.
    depth: (B, H, W, 1) - depth map of cam1, tensor
    intrinsics_cam1: (B, 3, 3) - intrinsics matrix of cam1, or (B, 3, 4)
    intrinsics_cam2: (B, 3, 3) - intrinsics matrix of cam2, or (B, 3, 4)
    pose_cam1_to_cam2: (B, 4, 4) - pose matrix from cam1 to cam2
    rgb_img_raw: (B, 3, H, W) - RGB image of cam1
    rgb_img_right_raw: (B, 3, H, W) - RGB image of cam2
    """
    lpix_in_r, ldepth_in_r = remap_pixel_coordinates(depth_cam1, intrinsics_cam1, intrinsics_cam2, pose_cam1_to_cam2)  # The left and right images have the same intrinsics
    # Show the feature correspondence between the left and right images based on the depth map
    # plot_feat_matching(rgb_img_raw[0], rgb_img_right_raw[0], lpix_in_r[0])

    # Calculate the photometric error for the correspondences
    img_h, img_w = rgb_img_raw.shape[-2], rgb_img_raw.shape[-1]
    rgbr_c2l = grid_sample(rgb_img_right_raw, lpix_in_r, mode='bilinear', padding_mode='zeros',
                           align_corners=True)
    mask_photometric_error = (lpix_in_r[..., 0] > 0) & (lpix_in_r[..., 0] < img_w) & (lpix_in_r[..., 1] > 0) & (
                lpix_in_r[..., 1] < img_h)
    mask_photometric_error = mask_photometric_error & (ldepth_in_r.squeeze(dim=-1) > 1e-3)  # (B, H, W)
    l1_photometric_error = torch.abs(rgbr_c2l - rgb_img_raw).norm(dim=1)  # (B, H, W)
    # Calculate the ssim error for the correspondences
    ssim_photometric_error = calculate_ssim_loss_batched(rgb_img_raw, rgb_img_right_raw, lpix_in_r,
                                                         mask_photometric_error, patch_size=7)  # (B, H, W)
    # print("-------------- l1_photometric_error[mask_photometric_error].mean(), ssim_photometric_error[mask_photometric_error].mean(): ", mask_photometric_error.sum(dim=(1, 2)), l1_photometric_error[mask_photometric_error].mean(), ssim_photometric_error[mask_photometric_error].mean())

    final_mask = mask_photometric_error & (l1_photometric_error < l1_thresh) & (ssim_photometric_error < ssim_thresh)  # (B, H, W)
    # print("============== final mask, mask_photometric_error: ", final_mask.sum(dim=(1, 2)), mask_photometric_error.sum(dim=(1, 2)), (l1_photometric_error < 0.8).sum(dim=(1, 2)), (ssim_photometric_error < 0.9).sum(dim=(1, 2)))
    print("============== Keep depth ratio: ", final_mask.sum(dim=(1, 2)) / mask_photometric_error.sum(dim=(1, 2)))
    return final_mask # (B, H, W)

if __name__== "__main__":

    from zoedepth.utils.config import get_config
    config = get_config("zoedepth_rgbt", "train", "ms2thermal")
    config["distributed"] = False
    config["batch_size"] = 1
    config["workers"] = 4
    config["aug"] = False # ERROR: it there is data augmentation, check the correctness of tpix_in_rgb
    # config["auto_scale_thermal"]=False


    loader = MS2DataLoader(config=config, mode='train', modality='rgbtd')
    sample_id=0
    for data in loader.data:
        
        # sample = {}
        # for k, v in data.items():
        #     sample[k] = v[0]

        # ### ERROR
        # sample_id+=1
        # if sample_id < 1500:
        #     continue

        sample = data

        # print("sample['rgb_image'].shape, sample['rgb_depth'].shape, sample['image'].shape, sample['depth'].shape: ", sample['rgb_image'].shape, sample['rgb_depth'].shape, sample['image'].shape, sample['depth'].shape)
        # print("-------- Shape of sample['intrinsics'], sample['rgb_intrinsics'], sample['pose_t2rgb']: ", sample['intrinsics'].shape, sample['rgb_intrinsics'].shape, sample['pose_t2rgb'].shape)
        # print("sample['intrinsics'], sample['rgb_intrinsics'], sample['pose_t2rgb']: ", sample['intrinsics'],
        #       sample['rgb_intrinsics'], sample['pose_t2rgb'])


        ######### Filter the depth of left RGB
        rgb_img_raw = sample['rgb_image'].cuda()# (B, 3, H, W)
        B = rgb_img_raw.shape[0]
        rgb_img_right_raw = sample['rgb_image_right'].cuda() # (B, 3, H, W)
        rgb_depth = sample['rgb_depth'].cuda() # (B, 1, H, W)
        rgb_depth = rgb_depth.permute(0,2,3,1).contiguous() # (B, H, W, 1)
        rgb_final_mask = get_lr_consistency_mask(rgb_depth, sample['rgb_intrinsics'].cuda(), sample['rgb_intrinsics'].cuda(), sample['pose_rgb_l2r'].cuda(), rgb_img_raw, rgb_img_right_raw,
                                                 l1_thresh=0.9, ssim_thresh=0.9)

        rgb_depth_filtered =sample['rgb_depth'].squeeze(dim=1).clone() # (B, 1, H, W) -> (B, H, W)
        rgb_depth_filtered[~rgb_final_mask] = 0.0

        # # plot the images
        # fig, axs = plt.subplots(2, 3, figsize=(20, 10))
        # rgb_img = sample['rgb_image'][0].cpu().numpy().transpose(1, 2, 0)
        #
        # axs[0, 0].imshow(rgb_img)
        # axs[0, 0].set_title('RGB Image')
        # axs[0, 0].axis('off')
        #
        # axs[0, 1].imshow(sample['rgb_depth'][0,0].cpu().numpy(), cmap='jet')
        # axs[0, 1].set_title('RGB Depth')
        # axs[0, 1].axis('off')
        #
        # rgb_depth = sample['rgb_depth'][0,0].cpu().numpy()
        # # depth_w = rgb_depth.shape[1]
        # # depth_h = rgb_depth.shape[0]
        # # rgb_img = cv2.resize(rgb_img, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)
        # vis_rgb_depth = visualize_rgb_depth(rgb_img, rgb_depth, min_depth=0.001, max_depth=80)
        # axs[0, 2].imshow(vis_rgb_depth)
        # axs[0, 2].set_title('RGB-Depth')
        # axs[0, 2].axis('off')
        #
        # rgb_img_right = sample['rgb_image_right'][0].cpu().numpy().transpose(1, 2, 0)
        # axs[1, 0].imshow(rgb_img_right)
        # axs[1, 0].set_title('Right RGB Image')
        # axs[1, 0].axis('off')
        #
        #
        # axs[1, 1].imshow(rgb_depth_filtered[0].cpu().numpy(), cmap='jet')
        # axs[1, 1].set_title('Filtered RGB Depth')
        # axs[1, 1].axis('off')
        #
        #
        # ## Visualize the depth label
        # rgb_depth_label = sample['rgb_depth'].squeeze(dim=1).clone() # (B, 1, H, W) -> (B, H, W)
        # original_depth_mask = rgb_depth_label > 0.01
        # rgb_depth_label[original_depth_mask] = 70.0
        # rgb_depth_label[rgb_final_mask] = 2.0
        #
        # vis_rgb_depth = visualize_rgb_depth( rgb_img, rgb_depth_label[0].cpu().numpy(), min_depth=0.001, max_depth=80)
        # axs[1, 2].imshow(vis_rgb_depth)
        # axs[1, 2].set_title('Label_RGB-Depth')
        # axs[1, 2].axis('off')
        #
        # plt.tight_layout()
        # plt.show()

        # plt.savefig('sample.png')
        # break

        ######### Filter the depth of left Thermal
        ther_img_raw = sample['image'].cuda()  # (B, 3, H, W)
        ther_img_right_raw = sample['image_right'].cuda()  # (B, 3, H, W)
        ther_depth = sample['depth'].cuda()  # (B, 1, H, W)
        ther_depth = ther_depth.permute(0, 2, 3, 1).contiguous()  # (B, H, W, 1)
        ther_final_mask = get_lr_consistency_mask(ther_depth, sample['intrinsics'].cuda(), sample['intrinsics'].cuda(),
                                                 sample['pose_ther_l2r'].cuda(), ther_img_raw, ther_img_right_raw, l1_thresh=1.65, ssim_thresh=0.75)

        ther_depth_filtered = sample['depth'].squeeze(dim=1).clone()  # (B, 1, H, W) -> (B, H, W)
        ther_depth_filtered[~ther_final_mask] = 0.0

        # # plot the images
        # fig, axs = plt.subplots(2, 3, figsize=(20, 10))
        # ther_img = sample['image'][0].cpu().numpy().transpose(1, 2, 0)
        #
        # axs[0, 0].imshow(ther_img)
        # axs[0, 0].set_title('Ther Image')
        # axs[0, 0].axis('off')
        #
        # axs[0, 1].imshow(sample['depth'][0,0].cpu().numpy(), cmap='jet')
        # axs[0, 1].set_title('Ther Depth')
        # axs[0, 1].axis('off')
        #
        # ther_depth = sample['depth'][0,0].cpu().numpy()
        # vis_ther_depth = visualize_rgb_depth(ther_img, ther_depth, min_depth=0.001, max_depth=80)
        # axs[0, 2].imshow(vis_ther_depth)
        # axs[0, 2].set_title('Ther-Depth')
        # axs[0, 2].axis('off')
        #
        # ther_img_right = sample['image_right'][0].cpu().numpy().transpose(1, 2, 0)
        # axs[1, 0].imshow(ther_img_right)
        # axs[1, 0].set_title('Right Ther Image')
        # axs[1, 0].axis('off')
        #
        # axs[1, 1].imshow(ther_depth_filtered[0].cpu().numpy(), cmap='jet')
        # axs[1, 1].set_title('Filtered Ther Depth')
        # axs[1, 1].axis('off')
        #
        # ## Visualize the depth label
        # ther_depth_label = sample['depth'].squeeze(dim=1).clone()  # (B, 1, H, W) -> (B, H, W)
        # original_depth_mask = ther_depth_label > 0.01
        # ther_depth_label[original_depth_mask] = 70.0
        # ther_depth_label[ther_final_mask] = 2.0
        #
        # vis_ther_depth = visualize_rgb_depth(ther_img, ther_depth_label[0].cpu().numpy(), min_depth=0.001, max_depth=80)
        # axs[1, 2].imshow(vis_ther_depth)
        # axs[1, 2].set_title('Label_Ther-Depth')
        # axs[1, 2].axis('off')
        #
        # plt.tight_layout()
        # plt.show()

        ### TODO: Save the rgb_depth_filtered and ther_depth_filtered
        for b in range(B):
            save_depth_filename = sample['depth_path'][b].replace('depth_filtered', 'depth_filtered_myrefine')
            save_rgb_depth_filename = sample['rgb_depth_path'][b].replace('depth_filtered', 'depth_filtered_myrefine')

            # Create the folder for data saving
            output_dir = os.path.dirname(save_depth_filename)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_dir = os.path.dirname(save_rgb_depth_filename)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            ther_depth_to_save = (ther_depth_filtered[b].numpy() * 256.0).astype(np.uint16)
            rgb_depth_to_save = (rgb_depth_filtered[b].numpy() * 256.0).astype(np.uint16)
            cv2.imwrite(save_depth_filename, ther_depth_to_save)
            cv2.imwrite(save_rgb_depth_filename, rgb_depth_to_save)
            print("Saving: ", save_depth_filename)

        # plt.savefig('sample.png')
        # break


