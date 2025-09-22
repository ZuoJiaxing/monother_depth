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

from zoedepth.data.preprocess import CropParams, get_white_border, get_black_border
import imageio
from path import Path


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode, **kwargs):
    return transforms.Compose([
        ToTensor(mode=mode, **kwargs)
    ])


class ThermalDepthDataLoader(object):
    def __init__(self, config, mode, device='cpu', transform=None, modality='td', **kwargs):
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

        check_choices('Modality', modality, ['td', 'rgbd'])
        print("Modality: ", modality)

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
                                   shuffle= self.train_sampler is None,
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

        elif mode == 'test':
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
    def __init__(self, config, mode, transform=None, is_for_online_eval=False, modality='td', **kwargs):
        """
        Modalidty: list of modalities to load. Options: 'td', 'rgbd', 'rgbtd', which denotes thermal+depth, rgb + depth, and rgb+thermal+thermaldepth_rgbdepth respectively
        """
        self.config = config
        self.modality = modality
        self.mode = mode

        check_choices('Modality', modality, ['td', 'rgbd'])
        check_choices('Mode', mode, ['train', 'online_eval', 'test'])
        if config.scene_type == 'indoor':
            if self.mode == 'train':
                folder_list_path = self.config.data_path+'/train_indoor.txt'
            elif self.mode == 'online_eval':
                if self.modality == 'rgbd':
                    folder_list_path = self.config.data_path + '/val_indoor_welllit.txt'
                else:
                    folder_list_path = self.config.data_path+'/val_indoor.txt'
            elif self.mode == 'test':
                if self.modality == 'rgbd':
                    folder_list_path = self.config.data_path + '/test_indoor_welllit.txt'
                else:
                    folder_list_path = self.config.data_path+'/test_indoor.txt'
        elif config.scene_type == 'outdoor':
            if self.mode == 'train':
                folder_list_path = self.config.data_path+'/train_outdoor.txt'
            elif self.mode == 'online_eval':
                folder_list_path = self.config.data_path+'/val_outdoor.txt'
            elif self.mode == 'test':
                folder_list_path = self.config.data_path+'/test_outdoor.txt'
        self.seq_folders = [os.path.join(self.config.data_path, folder[:-1]) for folder in open(folder_list_path)]

        # print("self.seq_folders: ", self.seq_folders)
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

                if self.modality == 'td':
                    intrinsics = np.genfromtxt(os.path.join(folder, 'cam_T.txt')).astype(np.float32).reshape((3, 3))
                    sample_name = {'image_path': imgs[i], 'depth_path': depth, 'focal': intrinsics[0,0]}
                elif self.modality == 'rgbd':
                    intrinsics = np.genfromtxt(os.path.join(folder, 'cam_RGB.txt')).astype(np.float32).reshape((3, 3))
                    sample_name = {'image_path': rgb, 'depth_path': rgb_depth, 'focal': intrinsics[0,0]}
                # elif self.modality == 'rgbtd':
                #     intrinsics = np.genfromtxt(os.path.join(folder, 'cam_T.txt')).astype(np.float32).reshape((3, 3))
                #     rgb_intrinsics = np.genfromtxt(os.path.join(folder, 'cam_RGB.txt')).astype(np.float32).reshape((3, 3))
                #     t2rgb_extrinsics = np.genfromtxt(os.path.join(folder, 'Tr_T2RGB.txt')).astype(np.float32).reshape((4, 4))
                #     sample_name = {'image_path': imgs[i], 'depth_path': depth, 'rgb_path': rgb, 'rgb_depth_path': rgb_depth,  'intrinsics': intrinsics, 'rgb_intrinsics': rgb_intrinsics, 't2rgb_extrinsics': t2rgb_extrinsics}

                self.filenames.append(sample_name)
        print("################# Length of RGBTDepthDataLoader: ", len(self.filenames))
        # TODO: save the sequence_name_set to a file

        self.transform = transform
        self.to_tensor = ToTensor(mode, modality=modality)
        self.is_for_online_eval = is_for_online_eval
        # self.reader = ImReader()

        rearr_thermal = self.config.get("rearr_thermal", False)
        if self.modality == 'td' and rearr_thermal:
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

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = sample_path['focal']
        sample = {}

        image_path = sample_path['image_path']
        depth_path = sample_path['depth_path']
        # rgb_path = sample_path['rgb_path']
        # rgb_depth = sample_path['rgb_depth_path']

        # print("for debug: ", idx, len(self.filenames), image_path, depth_path)

        if self.mode == 'train':
            image = None
            if self.modality == 'td':
                image = self.load_thermal_as_float(image_path)/2**14 # HW, [0-1]
            else: # Load rgb image
                image = self.load_rgb_as_float(image_path)
            depth_gt =  np.load(depth_path).astype(np.float32)

            # Resize the depth to the image size
            if depth_gt.shape[0] != image.shape[0] or depth_gt.shape[1] != image.shape[1]:
                depth_gt = cv2.resize(depth_gt, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            if self.modality == 'td' and self.config.auto_scale_thermal:
                image = self.rawthermal_autoScale(image)

            if self.config.do_random_rotate and (self.config.aug):
                if image.ndim == 3:
                    image = Image.fromarray((image*255).astype(np.uint8))
                else:
                    image = Image.fromarray(image)
                depth_gt = Image.fromarray(depth_gt)
                random_angle = (random.random() - 0.5) * 2 * self.config.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(
                    depth_gt, random_angle, flag=Image.NEAREST)

                image = np.asarray(image, dtype=np.float32)
                if image.ndim == 3:
                    image = image/255.0
                depth_gt = np.asarray(depth_gt, dtype=np.float32)

            if image.ndim == 2:
                image = np.expand_dims(image, axis=2) # HW -> HW1
            depth_gt = np.expand_dims(depth_gt, axis=2) # HW1

            if self.config.aug and (self.config.random_crop):
                image, depth_gt = self.random_crop(
                    image, depth_gt, self.config.input_height, self.config.input_width)
            
            if self.config.aug and self.config.random_translate:
                # print("Random Translation!")
                image, depth_gt = self.random_translate(image, depth_gt, self.config.max_translation)

            image, depth_gt = self.train_preprocess(image, depth_gt)
            mask = np.logical_and(depth_gt > self.config.min_depth,
                                  depth_gt < self.config.max_depth).squeeze()[None, ...]
            sample = {'image': image, 'depth': depth_gt, 'focal': focal,
                      'mask': mask, **sample}
        else:
            image = None
            if self.modality == 'td':
                image = self.load_thermal_as_float(image_path)/2**14 # HW, [0-1]
            else: # Load rgb image
                image = self.load_rgb_as_float(image_path)

            if self.modality == 'td' and self.config.auto_scale_thermal:
                image = self.rawthermal_autoScale(image)

            if image.ndim == 2:
                image = np.expand_dims(image, axis=2)  # HW -> HW1

            if (self.mode == 'online_eval' or self.mode == 'test'):
                has_valid_depth = False
                depth_gt = None
                try:
                    depth_gt = np.load(depth_path).astype(np.float32)
                    has_valid_depth = True
                except IOError:
                    depth_gt = None
                    # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    # Resize the depth to the image size
                    if depth_gt.shape[0] != image.shape[0] or depth_gt.shape[1] != image.shape[1]:
                        depth_gt = cv2.resize(depth_gt, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

                    depth_gt = np.expand_dims(depth_gt, axis=2) # HW1
                    mask = np.logical_and(depth_gt >= self.config.min_depth, depth_gt <= self.config.max_depth).squeeze()[None, ...]
                else:
                    mask = False
            sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth,
                      'image_path': image_path, 'depth_path': depth_path, 'mask': mask}
            # if (self.mode == 'online_eval' or self.mode == 'test'):
            #     sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth, 'image_path': image_path, 'depth_path': depth_path, 'mask': mask}
            # else:
            #     sample = {'image': image, 'focal': focal}

        if (self.mode == 'train') or ('has_valid_depth' in sample and sample['has_valid_depth']):
            mask = np.logical_and(depth_gt > self.config.min_depth, depth_gt < self.config.max_depth).squeeze()[None, ...]
            sample['mask'] = mask


        if self.transform:
            sample = self.transform(sample) # HWC -> CHW

        # Deal with tensors
        if self.thermal_transform:
            # print("-=-= for debug, before: ", type(sample["image"]), sample["image"].shape, sample["image"].min(), sample["image"].max())
            sample = self.thermal_transform(sample)
            # print("-=-= for debug, after: ", type(sample["image"]), sample["image"].shape, sample["image"].min(), sample["image"].max())

        if self.modality=='td' and self.config.dup_thermal3:
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

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]

        return img, depth
    
    def random_translate(self, img, depth, max_t=20):
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        p = self.config.translate_prob
        do_translate = random.random()
        if do_translate > p:
            return img, depth
        x = random.randint(-max_t, max_t)
        y = random.randint(-max_t, max_t)
        M = np.float32([[1, 0, x], [0, 1, y]])
        # print(img.shape, depth.shape)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        depth = cv2.warpAffine(depth, M, (depth.shape[1], depth.shape[0]))
        depth = depth.squeeze()[..., None]  # add channel dim back. Affine warp removes it
        # print("after", img.shape, depth.shape)
        return img, depth

    def train_preprocess(self, image, depth_gt):
        if self.config.aug:
            # Random flipping
            do_flip = random.random()
            if do_flip > 0.5:
                image = (image[:, ::-1, :]).copy()
                depth_gt = (depth_gt[:, ::-1, :]).copy()

            # Random gamma, brightness, color augmentation
            do_augment = random.random()
            if do_augment > 0.5:
                single_channel = image.shape[2] == 1
                image = self.augment_image(image, single_channel=single_channel)

        return image, depth_gt

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
    def __init__(self, mode, do_normalize=False, size=None, modality='td'):
        self.mode = mode
        if modality == 'td':
            self.normalize = transforms.Normalize(
                mean=[0.45], std=[0.225]) if do_normalize else nn.Identity() # For vivid thermal dataset
        elif modality == 'rgbd':
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if do_normalize else nn.Identity()
        self.size = size
        if size is not None:
            self.resize = transforms.Resize(size=size)
        else:
            self.resize = nn.Identity()

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)
        image = self.resize(image)

        # if self.mode == 'test':
        #     return {'image': image, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {**sample, 'image': image, 'depth': depth, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            image = self.resize(image)
            if has_valid_depth:
                depth = self.to_tensor(depth)
            return {**sample, 'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth,
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


if __name__== "__main__":

    from zoedepth.utils.config import get_config
    config = get_config("zoedepth", "train", "vivid_indoor")
    config["distributed"] = False
    config["batch_size"] = 1
    config["workers"] = 1
    config["aug"] = True
    data_loader = ThermalDepthDataLoader(config, "train", modality='rgbd').data # "online_eval"

    for batch_idx, sample in enumerate(data_loader):
        print("------------------------------------------------")
        print("batchid: ", batch_idx+1)
        print("sample.keys(): ", sample.keys())
        # print(sample["image"].shape, sample["depth"].shape, sample["focal"], sample["image_path"])

        # Visualize the depth and thermal image
        from PIL import Image
        import torchvision.transforms as transforms
        from zoedepth.utils.misc import colorize
        from zoedepth.utils.depth_utils import util_add_row,colored_depthmap

        _,_, h, w = sample["image"].shape
        resize_func = transforms.Resize((h, w), transforms.InterpolationMode.NEAREST)
        depth =  resize_func(sample["depth"]) # sample["depth"]
        # vis_depth = colorize(depth.squeeze().cpu().numpy(), 0, 10)
        vis_depth =  colored_depthmap(depth.squeeze().cpu().numpy(), 0, 7).astype(np.uint8)
        vis_img = 255.0*sample["image"].squeeze().cpu().numpy()
        vis_img = vis_img.transpose(1,2,0).astype(np.uint8)
        print("vis_img: : ", vis_img.shape, vis_img.min(), vis_img.max())
        print("vis_depth: : ", vis_depth.shape, vis_depth.min(), vis_depth.max())

        # vis_total = np.hstack((vis_img, vis_depth))
        # # vis_total = np.hstack((np.stack([vis_img,vis_img,vis_img], axis=2), vis_depth))
        # cv2.imshow("vis_total", vis_total)
        # cv2.waitKey()

        # For debug, check whether the image and depthmap are aligned
        halfw = int(w/2)
        vis_total = np.hstack((vis_img[:, :halfw,:], vis_depth[:, halfw:,:]))
        # vis_total = np.hstack((np.stack([vis_img,vis_img,vis_img], axis=2), vis_depth))
        cv2.imshow("vis_total", vis_total)
        cv2.waitKey()


