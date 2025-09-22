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

from zoedepth.utils.config import check_choices




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
                photometric_transforms=photometric_transforms,
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
                shuffle=(self.train_sampler is None),
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
            # thermal_depth_folder_path = os.path.join(proj_depth_path, 'thr', 'depth_filtered_myrefine')  # 'depth_filtered_myrefine', 'depth_filtered'
            # rgb_depth_folder_path = os.path.join(proj_depth_path, 'rgb', 'depth_filtered_myrefine')

            if self.mode == "train":
                thermal_depth_folder_path = os.path.join(proj_depth_path, 'thr', 'depth_filtered_myrefine') # 'depth_filtered_myrefine', 'depth_filtered'
                rgb_depth_folder_path = os.path.join(proj_depth_path, 'rgb', 'depth_filtered_myrefine')
            else:
                thermal_depth_folder_path = os.path.join(proj_depth_path, 'thr', 'depth_filtered')  # 'depth_filtered_myrefine', 'depth_filtered'
                rgb_depth_folder_path = os.path.join(proj_depth_path, 'rgb',  'depth_filtered')

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


        # Read data
        # thermal_img_raw = cv2.imread(image_path, -1)
        # rgb_img = cv2.imread(rgb_path, 1).astype(np.float32)
        # rgb_img = rgb_img/255.0  # (h, w, 3)
        # # rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) / 255.0 # (h, w, 3)  # This is wrong

        thermal_img_raw = self.load_thermal_as_float(image_path)
        rgb_img = self.load_rgb_as_float(rgb_path)
        
        # MS2 stores depth in metric scale depth * 256
        thermal_depth_gt = cv2.imread(depth_path, -1).astype(np.float32) / 256.0
        rgb_depth_gt = cv2.imread(rgb_depth_path, -1).astype(np.float32) / 256.0

        thermal_h, thermal_w = thermal_img_raw.shape
        rgb_h, rgb_w = rgb_img.shape[0], rgb_img.shape[1]

        ########################################################################################################################
        ### Resize the RGB images to the thermal image size
        if self.modality=="rgbtd" and (thermal_h != rgb_h or thermal_w != rgb_w):
            rgb_img = cv2.resize(rgb_img, (thermal_w, thermal_h), interpolation=cv2.INTER_LINEAR) # (1224, 384) -> (640, 256)  # cv2.INTER_NEAREST
            # rgb_depth_gt = cv2.resize(rgb_depth_gt, (thermal_w, thermal_h), interpolation=cv2.INTER_NEAREST) # (1224, 384) -> (640, 256) # cv2.INTER_NEAREST
            rgb_proj_matrix[0, :] = rgb_proj_matrix[0, :] * thermal_w/rgb_w
            rgb_proj_matrix[1, :] = rgb_proj_matrix[1, :] * thermal_h/rgb_h

        # Normalize the thermal image
        thermal_img = self.rawthermal_autoScale(thermal_img_raw, bits=16, minmax_norm=self.auto_scale_thermal, clahe=self.rearr_thermal)

        # Transform images only
        if self.photometric_transforms is not None:
            thermal_img = self.photometric_transforms(image=thermal_img)['image']
            rgb_img = self.photometric_transforms(image=rgb_img)['image']


        # Transform all data (images and depths) geometrically
        if self.geometric_transforms is not None:
            transformed_data = self.geometric_transforms(
                image=thermal_img, 
                depth=thermal_depth_gt, 
                rgb_image=rgb_img, 
                rgb_depth=rgb_depth_gt
            )

            thermal_img = transformed_data['image']
            thermal_depth_gt = transformed_data['depth']
            rgb_img = transformed_data['rgb_image']
            rgb_depth_gt = transformed_data['rgb_depth']

        
        # # Normalize thermal and rgb images to zero mean, unit std
        if self.do_normalize:
            thermal_img = self.thermal_normalize(image=thermal_img)['image']
            rgb_img = self.rgb_normalize(image=rgb_img)['image']

        # Get valid masks for depth
        thermal_mask = np.logical_and(thermal_depth_gt > self.min_depth,
                            thermal_depth_gt < self.max_depth).squeeze()
        rgb_mask = np.logical_and(rgb_depth_gt > self.min_depth,
                            rgb_depth_gt < self.max_depth).squeeze()

        # Convert to tensor
        thermal_img = ToTensorV2()(image=thermal_img)['image']
        rgb_img = ToTensorV2()(image=rgb_img)['image']
        thermal_depth_gt = ToTensorV2()(image=thermal_depth_gt)['image']
        rgb_depth_gt = ToTensorV2()(image=rgb_depth_gt)['image']
        thermal_mask = ToTensorV2()(image=thermal_mask)['image']
        rgb_mask = ToTensorV2()(image=rgb_mask)['image']

        # Duplicate the channeles of thermal image if it is single channel
        if thermal_img.shape[0] == 1:
            thermal_img = torch.cat([thermal_img]*3, dim=0)

        if self.modality == 'rgbtd':
            sample = {
                'dataset': 'ms2thermal',
                'image': thermal_img,
                'depth': thermal_depth_gt,

                'rgb_image': rgb_img,
                'rgb_depth': rgb_depth_gt,

                'has_valid_depth': True, # MS2 always has depth
                'intrinsics': thermal_proj_matrix,
                'rgb_intrinsics': rgb_proj_matrix,
                'pose_t2rgb': pose_t2rgb,
                # 'focal' : 0, # dummy value

                'image_path': image_path,
                'depth_path': depth_path,

                'mask': thermal_mask,
                'rgb_mask':rgb_mask
            }
        elif self.modality == 'rgbd':
            sample = {
                'dataset': 'ms2thermal',
                'image': rgb_img,
                'depth': rgb_depth_gt,
                # 'intrinsics': rgb_proj_matrix,
                'focal' : rgb_proj_matrix[0,0], # dummy value
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
                # 'intrinsics': thermal_proj_matrix,
                'focal': thermal_proj_matrix[0, 0],  # dummy value
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


if __name__== "__main__":

    from zoedepth.utils.config import get_config
    config = get_config("zoedepth_rgbt", "train", "ms2thermal")
    config["distributed"] = False
    config["batch_size"] = 1
    config["workers"] = 1
    config["aug"] = False # ERROR: it there is data augmentation, check the correctness of tpix_in_rgb


    loader = MS2DataLoader(config=config, mode='train', modality='rgbtd')
    for data in loader.data:
        
        sample = {}
        for k, v in data.items():
            sample[k] = v[0]

        # print("sample['rgb_image'].shape, sample['rgb_depth'].shape, sample['image'].shape, sample['depth'].shape: ", sample['rgb_image'].shape, sample['rgb_depth'].shape, sample['image'].shape, sample['depth'].shape)
        # print("-------- Shape of sample['intrinsics'], sample['rgb_intrinsics'], sample['pose_t2rgb']: ", sample['intrinsics'].shape, sample['rgb_intrinsics'].shape, sample['pose_t2rgb'].shape)
        # print("sample['intrinsics'], sample['rgb_intrinsics'], sample['pose_t2rgb']: ", sample['intrinsics'],
        #       sample['rgb_intrinsics'], sample['pose_t2rgb'])

        # plot the images
        fig, axs = plt.subplots(2, 3, figsize=(10, 10))
        thermal_img = sample['image'].squeeze().numpy()
        # axs[0, 0].imshow(thermal_img * 0.225 + 0.45, cmap='gray') # Denormalize
        axs[0, 0].imshow(thermal_img[0], cmap='gray')
        axs[0, 0].set_title('Thermal Image')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(sample['depth'].squeeze().numpy(), cmap='jet')
        axs[0, 1].set_title('Thermal Depth')
        axs[0, 1].axis('off')


        thermal_depth = sample['depth'].squeeze().numpy()
        # depth_w = thermal_depth.shape[1]
        # depth_h = thermal_depth.shape[0]
        # thermal_img = cv2.resize(thermal_img.transpose(1, 2, 0), (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)
        # print("thermal_img.shape: ", thermal_img.shape, thermal_depth.shape)
        vis_ther_depth = visualize_rgb_depth( thermal_img.transpose(1, 2, 0), thermal_depth, min_depth=0.001, max_depth=80)
        axs[0, 2].imshow(vis_ther_depth)
        axs[0, 2].set_title('Thermal-Depth')
        axs[0, 2].axis('off')
        
        rgb_img = sample['rgb_image'].squeeze().numpy().transpose(1, 2, 0)
        # rgb_img = rgb_img * np.array([0.229, 0.224, 0.225]).reshape(1,1,3) + np.array([0.485, 0.456, 0.406]).reshape(1,1,3) # Denormalize

        axs[1, 0].imshow(rgb_img)
        axs[1, 0].set_title('RGB Image')
        axs[1, 0].axis('off')

        axs[1, 1].imshow(sample['rgb_depth'].squeeze().numpy(), cmap='jet')
        axs[1, 1].set_title('RGB Depth')
        axs[1, 1].axis('off')

        rgb_depth = sample['rgb_depth'].squeeze().numpy()
        # depth_w = rgb_depth.shape[1]
        # depth_h = rgb_depth.shape[0]
        # rgb_img = cv2.resize(rgb_img, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)
        # print("rgb_img.shape: ", rgb_img.shape, rgb_depth.shape)
        vis_rgb_depth = visualize_rgb_depth( rgb_img, rgb_depth, min_depth=0.001, max_depth=80)
        axs[1, 2].imshow(vis_rgb_depth)
        axs[1, 2].set_title('RGB-Depth')
        axs[1, 2].axis('off')

        plt.show()

        # plt.savefig('sample.png')
        # break
