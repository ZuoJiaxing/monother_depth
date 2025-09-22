from __future__ import division
import torch
import random
import numpy as np
import cv2

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        image.sub_(self.mean).div_(self.std)
        return sample




class TensorThermalRearrange(object):
    def __init__(self, bin_num = 30, CLAHE_clip = 2, CLAHE_tilesize = 8):
        self.bins = bin_num
        self.CLAHE = cv2.createCLAHE(clipLimit=CLAHE_clip, tileGridSize=(CLAHE_tilesize,CLAHE_tilesize))
    def __call__(self, sample):
        image = sample['image']
        """Rearrange the thermal image to have a better contrast; image is in shape CHW"""
        hist = torch.histc(image, bins=self.bins)
        imgs_max = image.max()
        imgs_min = image.min()
        itv = (imgs_max - imgs_min)/self.bins
        total_num = hist.sum() 

        _,H,W = image.shape

        ## This preprocess can lead to some weird artifacts, so we comment it out.
        # mul_mask_ = torch.zeros((self.bins,H,W))
        # sub_mask_ = torch.zeros((self.bins,H,W))
        # subhist_new_min = imgs_min.clone()
        # for x in range(0,self.bins) :
        #     subhist = (image > imgs_min+itv*x) & (image <= imgs_min+itv*(x+1))
        #     if (subhist.sum() == 0):
        #         continue
        #     subhist_new_itv = hist[x]/total_num
        #     mul_mask_[x,...] = subhist * (subhist_new_itv / itv)
        #     sub_mask_[x,...] = subhist * (subhist_new_itv / itv * -(imgs_min+itv*x) + subhist_new_min)
        #     subhist_new_min += subhist_new_itv
        #
        # mul_mask = mul_mask_.sum(axis=0, keepdim=True).detach()
        # sub_mask = sub_mask_.sum(axis=0, keepdim=True).detach()
        # im_ = mul_mask*image + sub_mask

        im_ = image

        im_ = self.CLAHE.apply((im_.squeeze()*255).numpy().astype(np.uint8)).astype(np.float32)
        im_ = np.expand_dims(im_, axis=2)
        img_out = torch.from_numpy(np.transpose(im_/255., (2, 0, 1)))
        sample['image'] = img_out
        return sample