import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
util_cmap = plt.cm.viridis
import sys
from time import time
from torch.nn.functional import grid_sample

# from .depth_transforms import *

def output2depth(out, rep):
    def helper(x):
        if rep == 'depth':
            return x
        elif rep == 'inverse':
            return 1./F.threshold(x,1e-3,1e-3)
        elif rep == 'disparity':
            return 1./(10.0 * torch.sigmoid(x) + 1e-3)
        elif rep == 'log':
            return torch.exp(x)
        elif rep == 'log_inverse':
            #print(f"x: {x.max():.3f}, {x.min():.3f}, {x.mean():.3f}")
            return torch.exp(-x)
        else:
            assert False

    if type(out) == dict:
        if rep == 'depth':
            return out
        # Any dict key that does not contain "mask" or "cov", or "code_" is considered depth.
        # TODO: If any others come up (IDK what could though), add them here
        ret_dict = {}
        for k in out.keys():
            if not ("mask"  in k or "cov" in k or "code_" in k):
                ret_dict[k] = helper(out[k])
            else:
                ret_dict[k] = out[k]
        return ret_dict
    else:
        return helper(out)

def depth2output(x, rep):
    if rep == 'depth':
        return x
    elif rep == 'inverse':
        return 1. / F.threshold(x, 1e-3, 1e-3)
    elif rep == 'disparity':
        raise NotImplementedError("No inverse implemented for disparity")
    elif rep == 'log':
        return torch.log(torch.max(1e-3 * torch.ones_like(x), x))
    elif rep == 'log_inverse':
        return -torch.log(torch.max(1e-3 * torch.ones_like(x), x))
    else:
        assert False


def depthmap2rangemap(depthmap, Kcam):
    '''
        Transfer the depthmap into rangemap
        :param depthmap: (nviews, H, W)
        :param Kcam: (nviews, 3, 3), intrinsic matrix of the camera
        :return: rangemap (nviews,H, W)
    '''
    nviews, H, W = depthmap.shape
    # print("depthmap.shape: ", depthmap.shape)
    # print("Kcam: ", Kcam[0,:,:])
    device = depthmap.device
    dtype = depthmap.dtype
    Kcam_inv = Kcam.inverse()  # (numviews,3,3)
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))  # (H, W)
    grid_x = torch.stack([grid_x.to(dtype=dtype, device=device)]*nviews)
    grid_y = torch.stack([grid_y.to(dtype=dtype, device=device)]*nviews)
    grid_x = grid_x.float()*depthmap
    grid_y = grid_y.float()*depthmap
    xyz = torch.stack((grid_x, grid_y, depthmap), 3)  # (numviews, H, W, 3)
    xyz = xyz.view(nviews, -1, 3) # (numviews, HxW, 3)
    xyz = xyz.permute(0,2,1).contiguous() # (numviews, 3, HxW)
    # Distance from the center of the camera
    Kinv_p = Kcam_inv @ xyz  # (numviews,3,HxW)
    rangev2 = (Kinv_p * Kinv_p).sum(1)  # (numviews, HxW)
    rangev = rangev2.sqrt()  # (numviews, HxW)
    rangemap = rangev.view(nviews, H, W)
    return rangemap


def depthmap2rangemap_bs(depthmap, Kcam):
    '''
        Transfer the depthmap into rangemap
        :param depthmap: (nviews,B, H, W)
        :param Kcam: (nviews, B, 3, 3), intrinsic matrix of the camera
        :return: rangemap (nviews,B, H, W)
    '''
    nviews, B, H, W = depthmap.shape
    _K = Kcam[0,0]
    _K_inv = _K.inverse()
    grid_v, grid_u = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))  # (H, W)
    uv1 = torch.stack((grid_u, grid_v, torch.ones(H, W)), 2)  # (H, W, 3)
    uv1 = uv1.to(dtype=depthmap.dtype, device=depthmap.device)
    uv1 = uv1.unsqueeze(-1) # (H, W, 3, 1)
    Kinv_uv1 = _K_inv@uv1 # (H, W, 3, 1)
    Kinv_uv1_sq = Kinv_uv1.transpose(-2, -1)@Kinv_uv1 # (H, W, 1, 1)
    #  Jacob_r_z_ck = sqrt(uv1.T * Kinv.T * Kinv * uv1)
    Jacob_r_z_ck = Kinv_uv1_sq.squeeze().sqrt() # (H, W)
    Jacob_r_z_ck = Jacob_r_z_ck.unsqueeze(0).expand(B, -1, -1) # (bs, H, W)
    Jacob_r_z_ck = Jacob_r_z_ck.unsqueeze(0).expand(nviews, -1, -1, -1) # (numviews, bs, H, W)
    rangemap = Jacob_r_z_ck*depthmap
    return rangemap



def depthmap2pc(depthmap, depth_proj):
    '''
        Back project the depthmap into 3D point cloud
        :param depthmap: (H, W)
        :param depth_proj: (4, 4), K X T(W2C)
        :return: out_xyz: the 3D points in world frame
    '''
    H, W = depthmap.shape
    device = depthmap.device
    dtype = depthmap.dtype
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid_x = grid_x.to(dtype=dtype, device=device)*depthmap # (H, W)
    grid_y = grid_y.to(dtype=dtype, device=device)*depthmap # (H, W)
    xyz = torch.stack((grid_x, grid_y, depthmap), 2).float()  # W(x), H(y), 3
    xyz1 = torch.cat([xyz, torch.ones(H, W, 1).to(dtype=dtype, device=device)], dim=2) # (W, H, 4)
    xyz1 = xyz1.view(-1, 4)
    xyz1 = xyz1.permute(1,0).contiguous()
    xyz1 = depth_proj.inverse() @ xyz1 # (4, pts)
    xyz1 = xyz1.permute(1, 0).contiguous()
    out_xyz = xyz1[:,:3] # (pts, 3)
    # out_xyz = out_xyz[out_xyz[:,-1] > 0.2]
    return out_xyz

def depthmap2pc_bs(depthmap, depth_proj):
    '''
        Back project the depthmap into 3D point cloud
        :param depthmap: ( B, nviews, H, W)
        :param depth_proj: (B, nviews, 4, 4), K X T(W2C)
        :return: out_xyz: the 3D points in world frame
    '''
    B, N, H, W = depthmap.shape
    device = depthmap.device
    dtype = depthmap.dtype
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W)) # (H, W)
    grid_y = grid_y.unsqueeze(0).expand(N, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1, -1) #(B, N, H, W)
    grid_x = grid_x.unsqueeze(0).expand(N, -1, -1)
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1, -1) #(B, N, H, W)

    grid_x = grid_x.to(dtype=dtype, device=device)*depthmap # (B, N, H, W)
    grid_y = grid_y.to(dtype=dtype, device=device)*depthmap # (B, N, H, W)
    xyz = torch.stack((grid_x, grid_y, depthmap), -1).float()  # (B, N, H, W, 3), and the last dimension is [W(x) H(y) d]
    xyz1 = torch.cat([xyz, torch.ones(B, N, H, W, 1).to(dtype=dtype, device=device)], dim=4) # (B, N, H, W, 4)
    xyz1 = xyz1.view(B*N, -1, 4) # (B*N, pts, 4)
    xyz1 = xyz1.permute(0, 2, 1).contiguous()  # (B*N, 4, pts)

    depth_proj_inv = depth_proj.view(-1, 4, 4).inverse() # (B*N, 4, 4)
    xyz1 = depth_proj_inv @ xyz1 # (B*N, 4, pts)
    xyz1 = xyz1.permute(0, 2, 1).contiguous()  # (B*N, pts, 4)
    out_xyz = xyz1[:, :, :3] # (B*N, pts, 3)
    out_xyz = out_xyz.view(B, N, H*W, 3)
    out_xyz = out_xyz.view(B, -1, 3)
    return out_xyz

def imask_viz(imask):
    imask_cpu = (255 * np.squeeze(imask.cpu().numpy())).astype(np.uint8)
    imask_cpu = cv2.cvtColor(imask_cpu, cv2.COLOR_GRAY2RGB) # white = inlier, black = outlier
    #imask_cpu = (255 * cm(1-imask_cpu)[:,:,:3]).astype(np.uint8)
    return imask_cpu

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    depth_relative[depth_relative>1.0] = 1.0
    return 255 * util_cmap(depth_relative)[:,:,:3] # H, W, C


# def util_merge_into_row(rgb, depth_target, depth_pred, deppth_uncer, d_min, d_max):
#     if rgb is not None:
#         rgb = [rgb]
#         rgb = [np.transpose(img.cpu().numpy(), (1, 2, 0)) for img in rgb]
#         # BGR OpenCV to RGB python
#         rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 \
#                    else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in rgb]
#     else:
#         rgb = []
#
#     depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
#     depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())
#     depth_pred_uncer_cpu = np.squeeze(deppth_uncer.data.cpu().numpy())
#
#     # d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
#     # d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
#     depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
#     depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
#     depth_pred_uncer_col = colored_depthmap(depth_pred_uncer_cpu, d_min/8.0, d_max/8.0)
#     # * extracts list into the tuple
#     return (*rgb, depth_target_col, depth_pred_col, depth_pred_uncer_col)

def util_merge_into_row_photoerr(rgb0, rgb1, depth_pred=None, photometric_err=None, d_min=0.0, d_max=3.0):
    output = []
    if rgb0 is not None:
        rgb0 = [rgb0]
        rgb0 = [np.transpose(img.cpu().numpy(), (1, 2, 0)) for img in rgb0]
        # BGR OpenCV to RGB python
        rgb0 = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 \
                   else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in rgb0]
    else:
        rgb0 = []

    output.extend(rgb0)

    if rgb1 is not None:
        rgb1 = [rgb1]
        rgb1 = [np.transpose(img.cpu().numpy(), (1, 2, 0)) for img in rgb1]
        # BGR OpenCV to RGB python
        rgb1 = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 \
                   else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in rgb1]
    else:
        rgb1 = []

    output.extend(rgb1)

    if depth_pred is not None:
        depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())
        # d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
        # d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
        depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
        output.append(depth_pred_col)
    if photometric_err is not None:
        photometric_err_cpu = np.squeeze(photometric_err.data.cpu().numpy())
        photometric_err_col = colored_depthmap(photometric_err_cpu, 0.0, 12.0)
        output.append(photometric_err_col)

    # * extracts list into the tuple
    return tuple(output)

def util_merge_into_row(rgb, depth_target, depth_pred, depth_error=None, depth_pred_uncer=None, imask=None, d_min=0., d_max=3.):
    output = []
    if rgb is not None:
        rgb = [rgb]
        rgb = [np.transpose(img.cpu().numpy(), (1, 2, 0)) for img in rgb]
        # BGR OpenCV to RGB python
        rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 \
                   else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in rgb]
    else:
        rgb = []

    # Add imask as image if avail
    if imask is not None:
        rgb += [imask_viz(imask)]

    output.extend(rgb)

    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())
    # d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    # d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    output.append(depth_target_col)
    output.append(depth_pred_col)

    if depth_error is not None:
        depth_error_cpu = np.squeeze(depth_error.cpu().numpy())
        depth_error_col = colored_depthmap(depth_error_cpu, d_min, d_max)
        output.append(depth_error_col)

    if depth_pred_uncer is not None:
        depth_pred_uncer_cpu = np.squeeze(depth_pred_uncer.data.cpu().numpy())
        depth_pred_uncer_col = colored_depthmap(depth_pred_uncer_cpu, d_min, d_max)
        output.append(depth_pred_uncer_col)

    return tuple(output)

def util_add_row(img_merge, row):
    if type(row) == tuple:
        row_mat = np.hstack(row)
    else:
        row_mat = row
    if type(img_merge) == tuple:
        img_merge = np.hstack(img_merge)
    return np.vstack([img_merge, row_mat])


def util_save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)


def remap_pixel_coordinates(depth_map1, cam1_proj_matrix, cam2_proj_matrix, pose_cam1tocam2=None):
    '''
    depth_map1 is with shape (B, H, W, C) where B is batch size.
    cam1_proj_matrix, cam2_proj_matrix is with shape (B, 3, 3)  -> They might be the 3x4 matrix
    pose_cam1tocam2 is with shape (B, 4, 4)
    '''
    # Assumes inputs are in the shape (B, H, W, C) where B is batch size
    batch_size, height, width, ch = depth_map1.shape
    assert ch == 1, "Depth map should have only 1 channel"
    y_coords, x_coords = torch.meshgrid(torch.arange(height, device=depth_map1.device),
                                        torch.arange(width, device=depth_map1.device))
    ones = torch.ones((batch_size, height, width), device=depth_map1.device)

    # Convert depth map to 3D points in Cam1's coordinate system
    points_cam1 = depth_map1.repeat(1, 1, 1, 3) * torch.stack((x_coords, y_coords, ones[0]), dim=-1).unsqueeze(
        0).repeat(batch_size, 1, 1, 1)  # B, H, W, 3
    points_cam1 = points_cam1.view(batch_size, -1, 3)  # B, H*W, 3

    if cam1_proj_matrix.shape[-1] == 4: # cam1_proj_matrix is B,3x4
        points_cam1 = points_cam1 - cam1_proj_matrix[:, :, 3].unsqueeze(1)  # B, H*W, 3

    cam1_proj_matrix_inv = torch.inverse(cam1_proj_matrix[:, :, :3]) # B, 3, 3
    points_cam1 = torch.bmm(cam1_proj_matrix_inv, points_cam1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()  # B, H*W, 3

    # Apply relative pose transformation to move points to Cam2's coordinate system
    points_cam1_homo = torch.cat([points_cam1, ones.view(batch_size, -1, 1)], dim=-1)  # B, H*W, 4

    if pose_cam1tocam2 is None:
        transformed_points = points_cam1_homo
    else:
        transformed_points = torch.bmm(points_cam1_homo, pose_cam1tocam2.permute(0, 2, 1).contiguous())  # B, H*W, 4

    # Project transformed 3D points onto the image plane of Cam2
    if cam2_proj_matrix.shape[-1] ==3: # cam2_proj_matrix is B,3x3
        points_cam2_homogeneous = torch.bmm(transformed_points[:, :, :3], cam2_proj_matrix.permute(0, 2, 1).contiguous())  # B, H*W, 3
    else: # cam2_proj_matrix is B,3x4
        points_cam2_homogeneous = torch.bmm(transformed_points, cam2_proj_matrix.permute(0, 2, 1).contiguous())  # B, H*W, 3

    tmpdepth = points_cam2_homogeneous[:, :, 2]  # B, H*W

    invalid_mask = tmpdepth < 1e-2
    tmpdepth = torch.where(torch.abs(tmpdepth) < 1e-4, tmpdepth + 1e-6, tmpdepth)
    points_cam2 = points_cam2_homogeneous[:, :, :2] / tmpdepth.unsqueeze(2).repeat(1, 1, 2)  # B, H*W, 2
    points_cam2[invalid_mask] = 1e5  # Assign large pix to filter out the outliers

    pixel_coordinates_cam2 = points_cam2.view(batch_size, height, width, 2).to(torch.float32)  # B, H, W, 2

    depth_incam2 = points_cam2_homogeneous[:, :, 2]  # B, H*W
    depth_incam2[invalid_mask] = 0.0
    depth_incam2 = depth_incam2.view(batch_size, height, width).unsqueeze(3).to(torch.float32)  # B, H, W, 1

    return pixel_coordinates_cam2, depth_incam2

# bpix_in_a, bdepth_in_a, a_depth_c2b # All of them are in cam_a coordinate
def remap_depth_values(a_depth_c2b, bpix_in_a, cam1_proj_matrix, cam2_proj_matrix, pose_cam1tocam2=None):
    '''
    a_depth_c2b is with shape (B, H, W, 1) where B is batch size. -> depth value in cam_a, corresponding to the pixels in cam_b (cam2)
    bpix_in_a is with shape (B, H, W, 2)  -> pixel coordinates in cam_a, corresponding to the pixels in cam_b (cam2)
    cam1_proj_matrix, cam2_proj_matrix is with shape (B, 3, 3)  -> They might be the 3x4 matrix
    pose_cam1tocam2 is with shape (B, 4, 4)
    '''
    # Assumes inputs are in the shape (B, H, W, C) where B is batch size
    batch_size, height, width, ch = a_depth_c2b.shape
    assert ch == 1, "Depth map should have only 1 channel"
    ones = torch.ones((batch_size, height, width), device=a_depth_c2b.device)

    # Convert depth map to 3D points in Cam1's coordinate system
    points_cam1 = a_depth_c2b.repeat(1, 1, 1, 3) * torch.stack((bpix_in_a[..., 0], bpix_in_a[..., 1], ones), dim=-1)  # B, H, W, 3
    points_cam1 = points_cam1.view(batch_size, -1, 3)  # B, H*W, 3

    if cam1_proj_matrix.shape[-1] == 4: # cam1_proj_matrix is B,3x4
        points_cam1 = points_cam1 - cam1_proj_matrix[:, :, 3].unsqueeze(1)  # B, H*W, 3

    cam1_proj_matrix_inv = torch.inverse(cam1_proj_matrix[:, :, :3]) # B, 3, 3
    points_cam1 = torch.bmm(cam1_proj_matrix_inv, points_cam1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()  # B, H*W, 3

    # Apply relative pose transformation to move points to Cam2's coordinate system
    points_cam1_homo = torch.cat([points_cam1, ones.view(batch_size, -1, 1)], dim=-1)  # B, H*W, 4

    if pose_cam1tocam2 is None:
        transformed_points = points_cam1_homo
    else:
        transformed_points = torch.bmm(points_cam1_homo, pose_cam1tocam2.permute(0, 2, 1).contiguous())  # B, H*W, 4

    # Project transformed 3D points onto the image plane of Cam2
    if cam2_proj_matrix.shape[-1] ==3: # cam2_proj_matrix is B,3x3
        points_cam2_homogeneous = torch.bmm(transformed_points[:, :, :3], cam2_proj_matrix.permute(0, 2, 1).contiguous())  # B, H*W, 3
    else: # cam2_proj_matrix is B,3x4
        points_cam2_homogeneous = torch.bmm(transformed_points, cam2_proj_matrix.permute(0, 2, 1).contiguous())  # B, H*W, 3

    tmpdepth = points_cam2_homogeneous[:, :, 2]  # B, H*W

    invalid_mask = (tmpdepth < 1e-2) & (a_depth_c2b.view(batch_size, -1) < 1e-2)
    tmpdepth = torch.where(torch.abs(tmpdepth) < 1e-4, tmpdepth + 1e-6, tmpdepth)
    points_cam2 = points_cam2_homogeneous[:, :, :2] / tmpdepth.unsqueeze(2).repeat(1, 1, 2)  # B, H*W, 2
    points_cam2[invalid_mask] = 1e5  # Assign large pix to filter out the outliers

    pixel_coordinates_cam2 = points_cam2.view(batch_size, height, width, 2).to(torch.float32)  # B, H, W, 2

    depth_incam2 = points_cam2_homogeneous[:, :, 2]  # B, H*W
    depth_incam2[invalid_mask] = 0.0
    depth_incam2 = depth_incam2.view(batch_size, height, width).unsqueeze(3).to(torch.float32)  # B, H, W, 1

    return pixel_coordinates_cam2, depth_incam2


def generate_depth_map_with_occlusion(tpix_in_rgb, tdepth_in_rgb, image_height, image_width):
    """
    Generates depth map considering occlusion.

    tpix_in_rgb: (B, H, W, 2) - pixel coordinates in the new image
    tdepth_in_rgb: (B, H, W, 1) - depth values corresponding to the pixel coordinates
    """
    B = tpix_in_rgb.shape[0]
    assert B == tdepth_in_rgb.shape[0]

    # Create a tensor for the depth map initialized with infinity values
    depth_maps = torch.full((B, image_height, image_width), float('inf'), dtype=tdepth_in_rgb.dtype,
                            device=tdepth_in_rgb.device)

    # Convert the pixel coordinates to integer indices
    tpix_in_rgb_int = tpix_in_rgb.reshape(B, -1, 2).round().to(torch.int64)  # (B, N, 2)
    tdepth_in_rgb_flat = tdepth_in_rgb.reshape(B, -1)  # (B, N)

    # Create a mask for valid pixels
    valid_mask = (
            (tdepth_in_rgb_flat > 0.1).squeeze() &
            (tpix_in_rgb_int[..., 0] > 0) &
            (tpix_in_rgb_int[..., 1] > 0) &
            (tpix_in_rgb_int[..., 0] < image_width) &
            (tpix_in_rgb_int[..., 1] < image_height)
    )  # (B, N)

    for b in range(B):
        valid_indices = valid_mask[b]
        valid_tpix = tpix_in_rgb_int[b][valid_indices]
        valid_tdepth = tdepth_in_rgb_flat[b][valid_indices]

        # ### Take care of occlusion
        # for i in range(valid_tpix.shape[0]):
        #     y, x = valid_tpix[i, 1].item(), valid_tpix[i, 0].item()
        #     depth_maps[b, y, x] = min(depth_maps[b, y, x], valid_tdepth[i].item())

        ### Ignore the occulusion
        depth_maps[b, valid_tpix[:, 1], valid_tpix[:, 0]] = valid_tdepth

    # Replace infinity values with 0 for invalid depth entries
    depth_maps[depth_maps == float('inf')] = 0

    return depth_maps.unsqueeze(1)  # (B, 1, H, W)


def generate_fmap_frompix(tpix_in_rgb_, fmap_in_, image_height, image_width):
    """
    Generates depth map considering occlusion.

    tpix_in_rgb_: (B, H, W, 2) - pixel coordinates in the new image
    fmap_in_: (B, C, H, W) - depth values corresponding to the pixel coordinates
    """
    B = tpix_in_rgb_.shape[0]
    assert B == fmap_in_.shape[0]
    # C = fmap_in_.shape[1]

    tpix_in_rgb = tpix_in_rgb_.clone() # (B, H, W, 2)
    tpix_in_rgb = torch.stack([2*tpix_in_rgb[:,:,:,0]/(image_width - 1) - 1, 2*tpix_in_rgb[:,:,:,1]/ (image_height - 1) - 1 ], dim=-1)
    fmap_out = grid_sample(fmap_in_, tpix_in_rgb, mode='bilinear', padding_mode='zeros', align_corners=True)
    return fmap_out # (B, C, H, W)


def cosine_similarity(feature_map1, feature_map2):
    """
    Computes the cosine similarity between two feature maps.

    Args:
        feature_map1: Tensor of shape (B, C, H, W)
        feature_map2: Tensor of shape (B, C, H, W)

    Returns:
        similarity: Tensor of shape (B, H, W) containing the cosine similarity for each feature in the batch.
    """
    # Ensure the feature maps have the same shape
    assert feature_map1.shape == feature_map2.shape, "Feature maps must have the same shape"

    B, C, H, W = feature_map1.shape

    # Flatten the spatial dimensions
    feature_map1_flat = feature_map1.view(B, C, -1)  # Shape: (B, C, H*W)
    feature_map2_flat = feature_map2.view(B, C, -1)  # Shape: (B, C, H*W)

    # Compute the dot product along the channels
    dot_product = torch.sum(feature_map1_flat * feature_map2_flat, dim=1)  # Shape: (B, H*W)

    # Compute the L2 norms of the feature maps
    norm1 = torch.norm(feature_map1_flat, p=2, dim=1)  # Shape: (B, H*W)
    norm2 = torch.norm(feature_map2_flat, p=2, dim=1)  # Shape: (B, H*W)

    # Compute the cosine similarity
    similarity = dot_product / (norm1 * norm2 + 1e-8)  # Shape: (B, H*W)

    # Reshape the similarity to (B, H, W)
    similarity = similarity.view(B, 1, H, W)

    return similarity


# def standard_train_transform(rgb, depth, output_size,
#                                  pretransforms=[], posttransforms=[]):
#         height, width = output_size
#
#         # Check image
#         assert rgb.dtype == np.uint8
#         assert len(rgb.shape) >= 2
#
#         # Check depth
#         assert depth.dtype == np.float32
#         assert len(depth.shape) == 2
#
#         s = np.random.uniform(1.0, 1.5)  # random scaling
#         depth_np = depth / s  # IMPORTANT: fix the depth for the scale!!!
#         angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
#         do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
#
#
#         # Rotating and scale matrix
#         T = np.eye(3)
#         T[:2, :] = cv2.getRotationMatrix2D((width / 2. - .5, height / 2. - .5), angle, s)
#         if do_flip:
#             # Flip over x axis, shift 'width' to the right
#             Tf = np.eye(3)
#             Tf[0, 0] = -1
#             Tf[0, 2] = width
#             T = T @ Tf
#
#         T = T[:2, :]
#
#         # Perform 1st step of data augmentation (image and depth map).
#         # Note that depth images should use NN interp so that
#         # missing values are not used in interpolation
#         img_transform = Compose([t for t in pretransforms] + [
#             Resize(output_size, cv2.INTER_LINEAR),
#             WarpAffine(T, cv2.INTER_LINEAR),
#         ] + [t for t in posttransforms])
#         depth_transform = Compose([t for t in pretransforms] + [
#             Resize(output_size, cv2.INTER_NEAREST),
#             WarpAffine(T, cv2.INTER_NEAREST),
#         ] + [t for t in posttransforms])
#         rgb_np = img_transform(rgb)
#         rgb_np = np.asfarray(rgb_np, dtype='float') / 255
#         depth_np = depth_transform(depth_np)
#         return rgb_np, depth_np


def device_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time()

#
# def compute_depth_errors(gt, pred, max_depth=np.inf):
#     '''
#
#     '''
#     # pred = pred.cpu().numpy().squeeze()
#     # gt = gt.cpu().numpy().squeeze()
#
#     pred = pred.cpu().numpy()
#     gt = gt.cpu().numpy()
#
#     valid1 = gt >= 0.5
#     valid2 = gt <= max_depth
#     valid = valid1 & valid2
#     gt = gt[valid]
#     pred = pred[valid]
#
#     n_valid = np.float32(len(gt))
#     if n_valid == 0:
#         return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
#
#     differences = gt - pred
#     abs_differences = np.abs(differences)
#     squared_differences = np.square(differences)
#     abs_error = np.mean(abs_differences)
#     abs_relative_error = np.mean(abs_differences / gt)
#     abs_inverse_error = np.mean(np.abs(1 / gt - 1 / pred))
#     squared_relative_error = np.mean(squared_differences / gt)
#     rmse = np.sqrt(np.mean(squared_differences))
#     ratios = np.maximum(gt / pred, pred / gt)
#     n_valid = np.float32(len(ratios))
#     ratio_125 = np.count_nonzero(ratios < 1.25) / n_valid
#     ratio_125_2 = np.count_nonzero(ratios < 1.25 ** 2) / n_valid
#     ratio_125_3 = np.count_nonzero(ratios < 1.25 ** 3) / n_valid
#     return abs_error, abs_relative_error, abs_inverse_error, squared_relative_error, rmse, ratio_125, ratio_125_2, ratio_125_3
#
# def depth_eval_results(groundtruths, predictions, max_depth=np.inf):
#     '''
#     param: groundtruths, list[Tensor], [(B, H, W)]
#     param: predictions, list[Tensor], [(B, H, W)]
#     '''
#     if groundtruths is not None:
#         errors = []
#         for i, prediction in enumerate(predictions):
#             errors.append(compute_depth_errors(groundtruths[i], prediction, max_depth))
#
#         error_names = ['abs_error', 'abs_relative_error', 'abs_inverse_error',
#                        'squared_relative_error', 'rmse', 'ratio_125', 'ratio_125_2', 'ratio_125_3']
#
#         errors = np.array(errors)
#         mean_errors = np.nanmean(errors, 0)
#         print("{:>25}, {:>25}, {:>25}, {:>25}, {:>25}, {:>25}, {:>25}, {:>25}".format(*error_names))
#         print("{:25.4f}, {:25.4f}, {:25.4f}, {:25.4f}, {:25.4f}, {:25.4f}, {:25.4f}, {:25.4f}".format(*mean_errors))

