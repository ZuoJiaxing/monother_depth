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

# This file is partly inspired from BTS (https://github.com/cleinc/bts/blob/master/pytorch/bts_dataloader.py); author: Jin Han Lee

import itertools
import os
import random

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.utils.data.distributed

import sys
import os
sys.path.append(os.getcwd())
# print(sys.path)

from zoedepth.utils.easydict import EasyDict as edict
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob

from zoedepth.utils.config import change_dataset, check_choices
import zoedepth.data.thermal.thermal_custom_transforms as thermal_custom_transforms
import zoedepth.data.thermal.VIVID_sequence_folders as VIVID_sequence_dataset
import zoedepth.data.thermal.VIVID_validation_folders as VIVID_validation_dataset
from zoedepth.data.ms2thermal import get_ms2thermal_loader

from zoedepth.data.preprocess import CropParams, get_white_border, get_black_border
import imageio
import matplotlib.pyplot as plt
from path import Path


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode, **kwargs):
    return transforms.Compose([
        ToTensor(mode=mode, **kwargs)
    ])


class RGBTDepthDataLoader(object):
    def __init__(self, config, mode, device='cpu', transform=None, modality='rgbtd', **kwargs):
        """
        Data loader for depth datasets

        Args:
            config (dict): Config dictionary. Refer to utils/config.py
            mode (str): "train" or "online_eval"
            device (str, optional): Device to load the data on. Defaults to 'cpu'.
            transform (torchvision.transforms, optional): Transform to apply to the data. Defaults to None.
        """

        self.config = config
        img_size = self.config.get("img_size", None)
        img_size = img_size if self.config.get("do_input_resize", False) else None

        check_choices('Modality', modality, ['rgbtd', 'rgbd', 'td'])
        print("$$$$$$$$$$$$$$$$$$$$$ Modality: ", modality)

        if config.dataset == 'ms2thermal':
            self.data = get_ms2thermal_loader(config, mode, modality=modality, **kwargs)
            return


        if transform is None:
            transform = preprocessing_transforms(mode, do_normalize=False, size=img_size)

        if mode == 'train':
            self.training_samples = DataLoadPreprocess(config, mode, transform=transform, modality=modality, device=device)

            if config.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples,
                                   batch_size=config.batch_size,
                                   shuffle= (self.train_sampler is None), # NOTE
                                   num_workers=config.workers,
                                   pin_memory=True,
                                   persistent_workers=True,
                                #    prefetch_factor=2,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(config, mode, transform=transform, modality=modality)
            if config.distributed:  # redundant. here only for readability and to be more explicit
                # Give whole test set to all processes (and report evaluation only on one) regardless
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=kwargs.get("shuffle_test", False),
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler)

        elif mode == 'test' or mode == 'test_train':
            self.testing_samples = DataLoadPreprocess(config, mode, transform=transform, modality=modality)
            self.data = DataLoader(self.testing_samples,1, shuffle=False, num_workers=1)
        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


def repetitive_roundrobin(*iterables):
    """
    cycles through iterables but sample wise
    first yield first sample from first iterable then first sample from second iterable and so on
    then second sample from first iterable then second sample from second iterable and so on

    If one iterable is shorter than the others, it is repeated until all iterables are exhausted
    repetitive_roundrobin('ABC', 'D', 'EF') --> A D E B D F C D E
    """
    # Repetitive roundrobin
    iterables_ = [iter(it) for it in iterables]
    exhausted = [False] * len(iterables)
    while not all(exhausted):
        for i, it in enumerate(iterables_):
            try:
                yield next(it)
            except StopIteration:
                exhausted[i] = True
                iterables_[i] = itertools.cycle(iterables[i])
                # First elements may get repeated if one iterable is shorter than the others
                yield next(iterables_[i])


class RepetitiveRoundRobinDataLoader(object):
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders

    def __iter__(self):
        return repetitive_roundrobin(*self.dataloaders)

    def __len__(self):
        # First samples get repeated, thats why the plus one
        return len(self.dataloaders) * (max(len(dl) for dl in self.dataloaders) + 1)


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class CachedReader:
    def __init__(self, shared_dict=None):
        if shared_dict:
            self._cache = shared_dict
        else:
            self._cache = {}

    def open(self, fpath):
        im = self._cache.get(fpath, None)
        if im is None:
            im = self._cache[fpath] = Image.open(fpath)
        return im


class ImReader:
    def __init__(self):
        pass

    # @cache
    def open(self, fpath):
        return Image.open(fpath)


class DataLoadPreprocess(Dataset):
    def __init__(self, config, mode, transform=None, is_for_online_eval=False, modality='rgbtd', **kwargs):
        """
        Modalidty: list of modalities to load. Options: 'td', 'rgbd', 'rgbtd', which denotes thermal+depth, rgb + depth, and rgb+thermal+thermaldepth_rgbdepth respectively
        """
        self.config = config
        self.modality = modality
        self.mode = mode

        check_choices('Modality', modality, ['rgbtd'])
        check_choices('Mode', mode, ['train', 'online_eval', 'test', 'test_train'])
        if config.scene_type == 'indoor':
            if self.mode == 'train' or self.mode == "test_train":
                folder_list_path = self.config.data_path + '/train_indoor.txt'
            elif self.mode == 'online_eval':
                if self.modality != 'td':
                    folder_list_path = self.config.data_path + '/val_indoor_welllit.txt'
                else:
                    folder_list_path = self.config.data_path+'/val_indoor.txt'
            elif self.mode == 'test':
                folder_list_path = self.config.data_path + '/test_indoor.txt'
                # if self.modality != 'td':
                #     folder_list_path = self.config.data_path + '/test_indoor_welllit.txt'
                # else:
                #     folder_list_path = self.config.data_path+'/test_indoor.txt'
        elif config.scene_type == 'outdoor':
            if self.mode == 'train' or self.mode == "test_train":
                folder_list_path = self.config.data_path+'/train_outdoor.txt'
            elif self.mode == 'online_eval':
                folder_list_path = self.config.data_path+'/val_outdoor.txt'
            elif self.mode == 'test':
                folder_list_path = self.config.data_path+'/test_outdoor.txt'
        self.seq_folders = [os.path.join(self.config.data_path, folder[:-1]) for folder in open(folder_list_path)]

        print("################# Data loaded, self.seq_folders: ", self.seq_folders)

        self.filenames = []
        for folder in self.seq_folders:
            imgs = sorted(glob.glob(os.path.join(folder, "Thermal", "*.png")))
            for i in range(len(imgs)):
                filename = imgs[i].split('/')[-1]
                filename_without_extension = filename.split('.')[0]
                depth = os.path.join(folder, "Depth_T", (filename_without_extension + '.npy'))
                rgb = os.path.join(folder, "RGB", (filename_without_extension + '.png'))
                rgb_depth = os.path.join(folder, "Depth_RGB", (filename_without_extension + '.npy'))

                if self.modality == 'rgbtd':
                    thermal_intrinsics = np.genfromtxt(os.path.join(folder, 'cam_T.txt')).astype(np.float32).reshape((3, 3))
                    rgb_intrinsics = np.genfromtxt(os.path.join(folder, 'cam_RGB.txt')).astype(np.float32).reshape((3, 3))
                    t2rgb_extrinsics = np.genfromtxt(os.path.join(folder, 'Tr_T2RGB.txt')).astype(np.float32).reshape((4, 4))
                    sample_name = {'image_path': imgs[i], 'depth_path': depth, 'rgb_path': rgb, 'rgb_depth_path': rgb_depth,  'intrinsics': thermal_intrinsics, 'rgb_intrinsics': rgb_intrinsics, 't2rgb_extrinsics': t2rgb_extrinsics}
                else:
                    print("==================== ERROR: Modality not supported! ====================")
                self.filenames.append(sample_name)
        print("################# Length of RGBTDepthDataLoader: ", len(self.filenames))
        # TODO: save the sequence_name_set to a file

        self.transform = transform
        self.to_tensor = ToTensor(mode, modality=modality)
        self.is_for_online_eval = is_for_online_eval
        # self.reader = ImReader()

        rearr_thermal = self.config.get("rearr_thermal", False)
        if rearr_thermal:
            TenThrRearran = thermal_custom_transforms.TensorThermalRearrange(bin_num=config.rearrange_bin, CLAHE_clip=config.clahe_clip)
            # thermal_normalize = thermal_custom_transforms.Normalize(mean=0.45, std=0.225) # For vivid dataset; this seems to be wrong since it genetrate negative values
            self.thermal_transform = thermal_custom_transforms.Compose([TenThrRearran])  # [TenThrRearran, thermal_normalize]
        else:
            self.thermal_transform = None



    def postprocess(self, sample):
        return sample

    def load_thermal_as_float(self, path):
        return imageio.v2.imread(path).astype(np.float32)  # HW

    def load_rgb_as_float(self, path):
        image = Image.open(path)
        image = np.asarray(image, dtype=np.float32) / 255.0
        return image  # HW3

    def rawthermal_autoScale(self, thermal_data): # thermal_data [0, 1] or [0, 255]
        threshold_lo = np.percentile(thermal_data, 2)
        threshold_hi = np.percentile(thermal_data, 98)
        thermal_data[thermal_data < threshold_lo] = threshold_lo
        thermal_data[thermal_data > threshold_hi] = threshold_hi
        cv2.normalize(thermal_data, thermal_data, 0, 255, cv2.NORM_MINMAX)
        return  np.clip(thermal_data/255.0, 0, 1)

    def remap_pixel_coordinates(self, depth_map1, cam1_proj_matrix, cam2_proj_matrix, pose_cam1tocam2):
        # Step 1: Convert depth map to 3D points in Cam1's coordinate system
        height, width, _ = depth_map1.shape
        y_coords, x_coords = np.indices((height, width))
        points_cam1 = np.concatenate([depth_map1]*3, axis=-1) * np.stack((x_coords, y_coords, np.ones((height, width))), axis=-1) # HW3
        points_cam1 = np.linalg.inv(cam1_proj_matrix)@(points_cam1.reshape(-1, 3).T)
        points_cam1 = points_cam1.T # n3 (n=H*W)

        # Step 2: Apply relative pose transformation to move points to Cam2's coordinate system
        points_cam1_homo = np.concatenate([points_cam1, np.ones((height*width, 1))], axis=-1) # n4
        transformed_points = np.dot(points_cam1_homo, pose_cam1tocam2.T) # n4

        # Step 3: Project transformed 3D points onto the image plane of Cam2
        points_cam2_homogeneous = np.dot(transformed_points[:,:3], cam2_proj_matrix.T) # n3
        tmpdepth = points_cam2_homogeneous[:,2] # n
        invalid_mask = tmpdepth<(1e-2)
        tmpdepth[np.abs(tmpdepth)<1e-4] += 1e-3
        points_cam2 = points_cam2_homogeneous[:, :2] / np.stack([tmpdepth]*2, axis=-1) #n2
        points_cam2[invalid_mask, :] = 1e5 # Assign large pix to filter out the outliers
        pixel_coordinates_cam2 = points_cam2.reshape(height, width, 2).astype(np.float32) # HW2

        depth_map2 = points_cam2_homogeneous[:,2] # n
        depth_map2[invalid_mask] = 0.0
        depth_map2 = depth_map2.reshape(height, width).astype(np.float32)
        depth_map2 = np.expand_dims(depth_map2, axis=2) # HW1
        return pixel_coordinates_cam2, depth_map2

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        sample = {}

        image_path = sample_path['image_path']
        depth_path = sample_path['depth_path']
        rgb_path = sample_path['rgb_path']
        rgb_depth_path = sample_path['rgb_depth_path']

        thermal_proj_matrix = sample_path['intrinsics']
        focal = thermal_proj_matrix[0,0]

        # print("for debug: ", idx, len(self.filenames), image_path, depth_path)

        if self.mode == 'train':
            image = self.load_thermal_as_float(image_path)/2**14 # HW, [0-1]
            rgb_image = self.load_rgb_as_float(rgb_path)
            depth_gt =  np.load(depth_path).astype(np.float32)
            rgb_depth_gt = np.load(rgb_depth_path).astype(np.float32)

            # print("Shpae of image, depth_gt, rgb_image, rgb_depth_gt: ", image.shape, depth_gt.shape, rgb_image.shape, rgb_depth_gt.shape)

            # Resize the depth to the image size
            # assert image.shape[0] == rgb_image.shape[0] and image.shape[1] == rgb_image.shape[1]
            # depth_gt = cv2.resize(depth_gt, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            # rgb_depth_gt = cv2.resize(rgb_depth_gt, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            if self.config.auto_scale_thermal:
                image = self.rawthermal_autoScale(image)

            if self.config.do_random_rotate and (self.config.aug):
                print("####################### There are random rotate! ")
                image = Image.fromarray(image)
                depth_gt = Image.fromarray(depth_gt)
                random_angle = (random.random() - 0.5) * 2 * self.config.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

                rgb_image = Image.fromarray((rgb_image * 255).astype(np.uint8))
                rgb_depth_gt = Image.fromarray(rgb_depth_gt)
                rgb_image = self.rotate_image(rgb_image, random_angle)
                rgb_depth_gt = self.rotate_image(rgb_depth_gt, random_angle, flag=Image.NEAREST)

                image = np.asarray(image, dtype=np.float32)
                depth_gt = np.asarray(depth_gt, dtype=np.float32)
                rgb_image = np.asarray(rgb_image, dtype=np.float32)/255.0
                rgb_depth_gt = np.asarray(rgb_depth_gt, dtype=np.float32)

            if image.ndim == 2:
                image = np.expand_dims(image, axis=2) # HW -> HW1
            depth_gt = np.expand_dims(depth_gt, axis=2) # HW1
            rgb_depth_gt = np.expand_dims(rgb_depth_gt, axis=2) # HW1

            rgb_proj_matrix = sample_path['rgb_intrinsics']
            pose_t2rgb = sample_path['t2rgb_extrinsics']


            thermal_h, thermal_w = image.shape[0], image.shape[1]
            rgb_h, rgb_w = rgb_image.shape[0], rgb_image.shape[1]

            ########################################################################################################################
            ### Resize the RGB images to the thermal image size
            if self.modality == "rgbtd" and (thermal_h != rgb_h or thermal_w != rgb_w):
                rgb_image = cv2.resize(rgb_image, (thermal_w, thermal_h),
                                     interpolation=cv2.INTER_LINEAR)  # (1224, 384) -> (640, 256)  # cv2.INTER_NEAREST
                # rgb_depth_gt = cv2.resize(rgb_depth_gt, (thermal_w, thermal_h), interpolation=cv2.INTER_NEAREST) # (1224, 384) -> (640, 256) # cv2.INTER_NEAREST
                rgb_proj_matrix[0, :] = rgb_proj_matrix[0, :] * thermal_w / rgb_w
                rgb_proj_matrix[1, :] = rgb_proj_matrix[1, :] * thermal_h / rgb_h

            if self.config.aug and (self.config.random_crop):
                image, depth_gt, x, y = self.random_crop(image, depth_gt, self.config.input_height, self.config.input_width)
                rgb_image, rgb_depth_gt, _, _ = self.random_crop(rgb_image, rgb_depth_gt, self.config.input_height, self.config.input_width, x, y)
            
            if self.config.aug and self.config.random_translate:
                # print("Random Translation!")
                image, depth_gt, x, y = self.random_translate(image, depth_gt, self.config.max_translation)
                rgb_image, rgb_depth_gt, _, _ = self.random_translate(rgb_image, rgb_depth_gt, self.config.max_translation, x, y)

            image, depth_gt, rgb_image, rgb_depth_gt = self.train_preprocess(image, depth_gt, rgb_image, rgb_depth_gt)
            # Remove the border effects on the depth map
            black_border_params = get_black_border(depth_gt, tolerance=0.4, cut_off=100, level_diff_threshold=0.01, min_border=40)
            top, bottom, left, right = black_border_params.top, black_border_params.bottom, black_border_params.left, black_border_params.right
            depth_gt = np.copy(depth_gt)
            depth_gt[:top, :, :] = 0
            depth_gt[bottom:, :, :] = 0
            depth_gt[:, :left, :] = 0
            depth_gt[:, right:, :] = 0



            mask = np.logical_and(depth_gt > self.config.min_depth,
                                  depth_gt < self.config.max_depth).squeeze()[None, ...]
            rgb_mask = np.logical_and(rgb_depth_gt > self.config.min_depth,
                                  rgb_depth_gt < self.config.max_depth).squeeze()[None, ...]

            # Compute the warp from thermal to rgb
            # assert image.shape[0] == depth_gt.shape[0] and image.shape[1] == depth_gt.shape[1]
            # HW2, HW1
            # tpix_in_rgb, tdepth_in_rgb = self.remap_pixel_coordinates(depth_gt, thermal_proj_matrix, rgb_proj_matrix, pose_t2rgb)

            # Visualize the warp
            sample = {'image': image, 'depth': depth_gt, 'rgb_image': rgb_image, 'rgb_depth': rgb_depth_gt, # 'tpix_in_rgb': tpix_in_rgb, 'tdepth_in_rgb': tdepth_in_rgb,
                      'intrinsics': thermal_proj_matrix, 'rgb_intrinsics': rgb_proj_matrix, 'pose_t2rgb': pose_t2rgb,
                      'focal': focal, 'mask': mask, 'rgb_mask':rgb_mask, **sample} # Rocky: as I have chekced, the focal is not used.
        else:
            image = self.load_thermal_as_float(image_path)/2**14 # HW, [0-1]
            rgb_image = self.load_rgb_as_float(rgb_path)


            # Resize the depth to the image size
            assert image.shape[0] == rgb_image.shape[0] and image.shape[1] == rgb_image.shape[1]


            if self.config.auto_scale_thermal:
                image = self.rawthermal_autoScale(image)

            if image.ndim == 2:
                image = np.expand_dims(image, axis=2)  # HW -> HW1


            thermal_h, thermal_w = image.shape[0], image.shape[1]
            rgb_h, rgb_w = rgb_image.shape[0], rgb_image.shape[1]

            rgb_proj_matrix = sample_path['rgb_intrinsics']
            pose_t2rgb = sample_path['t2rgb_extrinsics']

            ########################################################################################################################
            ### Resize the RGB images to the thermal image size
            if self.modality == "rgbtd" and (thermal_h != rgb_h or thermal_w != rgb_w):
                rgb_image = cv2.resize(rgb_image, (thermal_w, thermal_h),
                                     interpolation=cv2.INTER_LINEAR)  # (1224, 384) -> (640, 256)  # cv2.INTER_NEAREST
                # rgb_depth_gt = cv2.resize(rgb_depth_gt, (thermal_w, thermal_h), interpolation=cv2.INTER_NEAREST) # (1224, 384) -> (640, 256) # cv2.INTER_NEAREST
                rgb_proj_matrix[0, :] = rgb_proj_matrix[0, :] * thermal_w / rgb_w
                rgb_proj_matrix[1, :] = rgb_proj_matrix[1, :] * thermal_h / rgb_h


            if (self.mode == 'online_eval' or self.mode == 'test' or self.mode == 'test_train'):
                has_valid_depth = False
                depth_gt = None
                rgb_depth_gt = None
                mask = False
                rgb_mask = False
                tpix_in_rgb, tdepth_in_rgb = None, None

                try:
                    depth_gt = np.load(depth_path).astype(np.float32)
                    rgb_depth_gt = np.load(rgb_depth_path).astype(np.float32)
                    # depth_gt = cv2.resize(depth_gt, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # rgb_depth_gt = cv2.resize(rgb_depth_gt, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    has_valid_depth = True
                except IOError:
                    depth_gt = None
                    rgb_depth_gt = None
                    # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    depth_gt = np.expand_dims(depth_gt, axis=2)  # HW1
                    rgb_depth_gt = np.expand_dims(rgb_depth_gt, axis=2)  # HW1

                    # Remove the border effects on the depth map
                    black_border_params = get_black_border(depth_gt, tolerance=0.4, cut_off=100,
                                                           level_diff_threshold=0.01, min_border=40)
                    top, bottom, left, right = black_border_params.top, black_border_params.bottom, black_border_params.left, black_border_params.right
                    depth_gt = np.copy(depth_gt)
                    depth_gt[:top, :, :] = 0
                    depth_gt[bottom:, :, :] = 0
                    depth_gt[:, :left, :] = 0
                    depth_gt[:, right:, :] = 0

                    mask = np.logical_and(depth_gt > self.config.min_depth, depth_gt < self.config.max_depth).squeeze()[None, ...]
                    rgb_mask = np.logical_and(rgb_depth_gt > self.config.min_depth, rgb_depth_gt < self.config.max_depth).squeeze()[None, ...]

                    # Compute the warp from thermal to rgb
                    # assert image.shape[0] == depth_gt.shape[0] and image.shape[1] == depth_gt.shape[1]
                    # HW2, HW1
                    # tpix_in_rgb, tdepth_in_rgb = self.remap_pixel_coordinates(depth_gt, thermal_proj_matrix, rgb_proj_matrix, pose_t2rgb)
                else:
                    mask = False
                    rgb_mask = False
                sample = {'image': image, 'depth': depth_gt, 'rgb_image': rgb_image, 'rgb_depth': rgb_depth_gt, 'has_valid_depth': has_valid_depth, # 'tpix_in_rgb': tpix_in_rgb, 'tdepth_in_rgb': tdepth_in_rgb,
                          'intrinsics': thermal_proj_matrix, 'rgb_intrinsics': rgb_proj_matrix, 'pose_t2rgb': pose_t2rgb,
                          'image_path': image_path, 'depth_path': depth_path, 'focal': focal, 'mask': mask, 'rgb_mask':rgb_mask} # Rocky: as I have chekced, the focal is not used.

        # if (self.mode == 'train') or ('has_valid_depth' in sample and sample['has_valid_depth']):
        #     mask = np.logical_and(depth_gt > self.config.min_depth, depth_gt < self.config.max_depth).squeeze()[None, ...]
        #     sample['mask'] = mask


        if self.transform:
            sample = self.transform(sample) # HWC -> CHW

        # Deal with tensors
        if self.thermal_transform:
            # print("-=-= for debug, before: ", type(sample["image"]), sample["image"].shape, sample["image"].min(), sample["image"].max())
            sample = self.thermal_transform(sample)
            # print("-=-= for debug, after: ", type(sample["image"]), sample["image"].shape, sample["image"].min(), sample["image"].max())

        if sample['image'].shape[0] == 1 and self.config.dup_thermal3:
            sample['image'] = torch.cat([sample['image']]*3, dim=0)

        sample = self.postprocess(sample)
        sample['dataset'] = self.config.dataset
        sample = {**sample,  'image_path': image_path, 'depth_path': depth_path}


        # print("-=-= for debug, image: ", type(sample["image"]), sample["image"].shape, sample["image"].min(), sample["image"].max())
        # print("-=-= for debug, depth_gt: ", type(sample["depth"]), sample["depth"].shape, sample["depth"].min(), sample["depth"].max())
        # print("===================== For debug, image.shape, depth.shape: ", sample["image"].shape, sample["depth"].shape)

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width, x=None, y=None):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        if (x is None) or (y is None):
            x = random.randint(0, img.shape[1] - width)
            y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]

        return img, depth, x, y
    
    def random_translate(self, img, depth, max_t=20, x=None, y=None):
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        p = self.config.translate_prob
        do_translate = random.random()
        if do_translate > p:
            return img, depth
        if (x is None) or (y is None):
            x = random.randint(-max_t, max_t)
            y = random.randint(-max_t, max_t)
        M = np.float32([[1, 0, x], [0, 1, y]])
        # print(img.shape, depth.shape)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        depth = cv2.warpAffine(depth, M, (depth.shape[1], depth.shape[0]))
        depth = depth.squeeze()[..., None]  # add channel dim back. Affine warp removes it
        # print("after", img.shape, depth.shape)
        return img, depth, x, y

    def train_preprocess(self, image, depth_gt, rgb_image, rgb_depth_gt):
        if self.config.aug:
            # Random flipping
            do_flip = random.random()
            if do_flip > 0.5:
                image = (image[:, ::-1, :]).copy()
                depth_gt = (depth_gt[:, ::-1, :]).copy()
                rgb_image = (rgb_image[:, ::-1, :]).copy()
                rgb_depth_gt = (rgb_depth_gt[:, ::-1, :]).copy()

            # Random gamma, brightness, color augmentation
            do_augment = random.random()
            if do_augment > 0.5:
                image = self.augment_image(image, single_channel=True)
                rgb_image = self.augment_image(rgb_image, single_channel=False)

        return image, depth_gt, rgb_image, rgb_depth_gt

    def augment_image(self, image, single_channel=False):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.config.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=1) if single_channel else np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(len(colors))], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)
        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode, do_normalize=False, size=None, modality='rgbtd'):
        self.mode = mode
        self.normalize_thermal = transforms.Normalize(
            mean=[0.45], std=[0.225]) if do_normalize else nn.Identity() # For vivid thermal dataset
        self.normalize_rgb = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if do_normalize else nn.Identity()
        self.size = size
        if size is not None:
            self.resize = transforms.Resize(size=size)
        else:
            self.resize = nn.Identity()

    def __call__(self, sample):
        image, depth, focal = sample['image'], sample['depth'], sample['focal']
        rgb, rgb_depth = sample['rgb_image'], sample['rgb_depth']
        image = self.to_tensor(image)
        image = self.normalize_thermal(image)
        image = self.resize(image)
        rgb = self.to_tensor(rgb)
        rgb = self.normalize_rgb(rgb)
        rgb = self.resize(rgb)


        if self.mode == 'train':
            depth = self.to_tensor(depth)
            rgb_depth = self.to_tensor(rgb_depth)
            # tpix_in_rgb = self.to_tensor(sample['tpix_in_rgb']) # (2, H, W)
            # tdepth_in_rgb = self.to_tensor(sample['tdepth_in_rgb'])
            return {**sample, 'image': image, 'depth': depth, 'rgb_image': rgb, 'rgb_depth': rgb_depth,
                    #'tpix_in_rgb': tpix_in_rgb, 'tdepth_in_rgb': tdepth_in_rgb,
                    'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            image = self.resize(image)
            if has_valid_depth:
                depth = self.to_tensor(depth)
                rgb_depth = self.to_tensor(rgb_depth)
                # tpix_in_rgb = self.to_tensor(sample['tpix_in_rgb'])
                # tdepth_in_rgb = self.to_tensor(sample['tdepth_in_rgb'])
            return {**sample, 'image': image, 'depth': depth,  'rgb_image': rgb, 'rgb_depth': rgb_depth,
                    'focal': focal, 'has_valid_depth': has_valid_depth,
                    # 'tpix_in_rgb': tpix_in_rgb, 'tdepth_in_rgb': tdepth_in_rgb,
                    'image_path': sample['image_path'], 'depth_path': sample['depth_path']}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1))) # CHW
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img

def generate_depth_map(tpix_in_rgb, tdepth_in_rgb, image_height, image_width):
    """
    Note that this implementation is not optimal, since we do not consider the occulusion here and we round the pixel coordinates to the nearest integer.
    tpix_in_rgb: (B, 2, H, W)
    tdepth_in_rgb: (B, 1, H, W)
    """
    # Create an empty tensor for the depth map
    B = tpix_in_rgb.shape[0]
    assert B == tdepth_in_rgb.shape[0]
    depth_maps = []

    for i in range(B):
        depth_map = torch.zeros((image_height, image_width), dtype=tdepth_in_rgb.dtype, device=tdepth_in_rgb.device)
        # Convert the pixel coordinates to integer indices; Note that this is not perfect as the coordinates are rounded
        tpix_in_rgb_int = (tpix_in_rgb[i].permute(1, 2, 0).contiguous().reshape(-1, 2).round().to(torch.int64)) # (N, 2)
        tdepth_in_rgb = tdepth_in_rgb[i].permute(1, 2, 0).contiguous().reshape(-1) # (N)
        valid_mask = (tdepth_in_rgb > 0).squeeze() & (tpix_in_rgb_int[:, 0]>0 ) & (tpix_in_rgb_int[:, 1]>0) & (tpix_in_rgb_int[:, 0]<image_width) & (tpix_in_rgb_int[:, 1]<image_height)
        tpix_in_rgb_int = tpix_in_rgb_int[valid_mask] # (N,2)
        tdepth_in_rgb = tdepth_in_rgb[valid_mask] # (N,1)
        # Assign depth values to the corresponding pixel coordinates in the depth map
        depth_map[tpix_in_rgb_int[:, 1], tpix_in_rgb_int[:, 0]] = tdepth_in_rgb
        depth_maps.append(depth_map.unsqueeze(0))
    return torch.stack(depth_maps, dim=0) # (B, 1, H, W)


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
    # For visualization of pointcloud
    from zoedepth.utils.depth_utils import depthmap2pc
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    #================
    from torch.nn.functional import grid_sample
    from zoedepth.utils.config import get_config
    config = get_config("zoedepth_rgbt", "eval", "vivid_outdoor")
    config["distributed"] = False
    config["batch_size"] = 1
    config["workers"] = 1
    config["aug"] = True
    data_loader = RGBTDepthDataLoader(config, "test", modality='rgbtd').data # "online_eval"

    for batch_idx, sample_ in enumerate(data_loader):
        print("------------------------------------------------")
        print("batchid: ", batch_idx+1)
        # print("sample.keys(): ", sample.keys())
        print(sample_["image"].shape, sample_["depth"].shape, sample_["rgb_image"].shape, sample_["rgb_depth"].shape)
        print("mean, min and max in depth data: ", sample_["depth"].mean(), sample_["depth"].min(), sample_["depth"].max())

        sample = {}
        for k, v in sample_.items():
            sample[k] = v[0]

        # plot the images
        fig, axs = plt.subplots(2, 3, figsize=(10, 10))
        thermal_img = sample['image'].squeeze().numpy().transpose(1, 2, 0)
        # axs[0, 0].imshow(thermal_img * 0.225 + 0.45, cmap='gray') # Denormalize
        axs[0, 0].imshow(thermal_img)
        axs[0, 0].set_title('Thermal Image')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(sample['depth'].squeeze().numpy(), cmap='jet')
        axs[0, 1].set_title('Thermal Depth')
        axs[0, 1].axis('off')

        thermal_depth = sample['depth'].squeeze().numpy()
        thermal_img = cv2.resize(thermal_img, (thermal_depth.shape[1], thermal_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        print("thermal_img.shape: ", thermal_img.shape, thermal_depth.shape)
        vis_ther_depth = visualize_rgb_depth(thermal_img, thermal_depth, min_depth=config.min_depth, max_depth=config.max_depth)
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
        rgb_img = cv2.resize(rgb_img, (rgb_depth.shape[1], rgb_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        print("rgb_img.shape: ", rgb_img.shape, rgb_depth.shape)
        vis_rgb_depth = visualize_rgb_depth(rgb_img, rgb_depth, min_depth=config.min_depth, max_depth=config.max_depth)
        axs[1, 2].imshow(vis_rgb_depth)
        axs[1, 2].set_title('RGB-Depth')
        axs[1, 2].axis('off')

        plt.show()


        # # vis_total = np.hstack((vis_img[:, :half_w, :], vis_depth[:, half_w:, :], vis_img_rgb[:, :half_w, :], vis_depth_rgb[:, half_w:, :]))
        # # vis_total = np.hstack((vis_img[:, :half_w, :],  vis_img_rgb[:, half_w:, :], vis_depth[:, :half_w, :], vis_depth_rgb[:, half_w:, :]))
        # cv2.imshow("vis_total", vis_total)
        # cv2.waitKey()
        #
        # ### Visualize to check the correctness of the tpix_in_rgb
        # # print("--------------- depth_rgb.shape, sample['tpix_in_rgb'].shape: ", depth_rgb.shape, sample['tpix_in_rgb'].shape)
        # tpix_in_rgb = sample['tpix_in_rgb'].clone() # (B, 2, H, W)
        # tpix_in_rgb = tpix_in_rgb.permute(0, 2, 3, 1).contiguous() # (B, H, W, 2)
        # # tpix_in_rgb = tpix_in_rgb.view(b, -1, 2).unsqueeze(dim=1) # (B, 1, H*W, 2)
        # tpix_in_rgb = torch.stack([2*tpix_in_rgb[:,:,:,0]/(w - 1) - 1, 2*tpix_in_rgb[:,:,:,1]/ (h - 1) - 1 ], dim=-1)
        #
        # # print("--------------------- (sample['rgb_image']).dtype, (tpix_in_rgb).dtype: ", (sample['rgb_image']).dtype, (tpix_in_rgb).dtype)
        #
        # rgb_warpped_from_t = grid_sample(sample['rgb_image'], tpix_in_rgb, mode='bilinear', padding_mode='zeros', align_corners=True)
        # vis_rgb_warpped_from_t = rgb_warpped_from_t.view(3, h, w).squeeze().cpu().numpy()
        # # print("---------------------- vis_rgb_warpped_from_t.shape: ", vis_rgb_warpped_from_t.shape)
        # vis_rgb_warpped_from_t = (255.0 * vis_rgb_warpped_from_t.transpose(1, 2, 0)).astype(np.uint8)
        #
        # # Note that, this way to compute depth is not correct. It is just for visualization.
        # depth_warpped_from_t = grid_sample(sample['rgb_depth'], tpix_in_rgb, mode='nearest', padding_mode='zeros', align_corners=True)
        # depth_warpped_from_t = depth_warpped_from_t.view(1, h, w)
        # vis_depth_warpped_from_t = colored_depthmap(depth_warpped_from_t.squeeze().cpu().numpy(), 0, 7).astype(np.uint8)
        #
        # vis_check_warp = np.hstack((vis_img, vis_rgb_warpped_from_t, vis_depth, vis_depth_warpped_from_t))
        # cv2.imshow("thermal-wrgb-depth-wdepth", vis_check_warp)
        # vis_check_warp2 = np.hstack((vis_img[:, :half_w, :], vis_rgb_warpped_from_t[:, half_w:, :], vis_depth[:, :half_w, :], vis_depth_warpped_from_t[:, half_w:, :]))
        # cv2.imshow("half : thermal-wrgb-depth-wdepth", vis_check_warp2)
        # cv2.waitKey()
        #
        # # ##  This way to compute depth is correct, but not perfect, since we do not consider the occulusion.
        # # depth_warpped_from_t = generate_depth_map(sample['tpix_in_rgb'], sample['tdepth_in_rgb'], h, w) # (B, 1, H, W)
        # # # # For debug, check the proximity of the depth_warpped_from_t and rgb_depth_gt
        # # # valid_depth_mask = (depth_warpped_from_t>1e-3) & (sample['rgb_depth']>1e-3)
        # # # offset = (depth_warpped_from_t - sample['rgb_depth']).abs()
        # # # offset = offset[valid_depth_mask]
        # #
        # # vis_depth_warpped_from_t =  colored_depthmap(depth_warpped_from_t.squeeze().cpu().numpy(), 0, 7).astype(np.uint8)
        # # vis_check_warp = np.hstack((vis_img, vis_img_rgb, vis_depth_rgb, vis_depth_warpped_from_t))
        # # cv2.imshow("thermal-wrgb-rgbdepth-wdepth", vis_check_warp)
        # # vis_check_warp2 = np.hstack((vis_img_rgb, vis_depth_rgb[:, :half_w, :], vis_depth_warpped_from_t[:, half_w:, :]))
        # # cv2.imshow("half : rgb-rgbdepth-wdepth", vis_check_warp2)
        # # cv2.waitKey()

        # # Visualize the point cloud
        # rgb_intr = torch.eye(4).to(sample['rgb_intrinsics'].device)
        # rgb_intr[:3, :3] = sample['rgb_intrinsics'][0]
        # pointcloud = depthmap2pc(depth_rgb.squeeze(), rgb_intr).cpu().numpy()
        # indices = np.random.choice(pointcloud.shape[0], pointcloud.shape[0]//20, replace=False)
        # pointcloud = pointcloud[indices]
        #
        # # Create a 3D scatter plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], c='b', marker='o', s=1)
        # plt.show()




