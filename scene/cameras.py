#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from PIL import Image
from utils.general_utils import PILtoTensor

#old camera class
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, head_mask, mouth_mask,
                 exp_param, eyes_pose, eyelids, jaw_pose,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.head_mask = head_mask.to(self.data_device)
        self.mouth_mask = mouth_mask.to(self.data_device)
        self.exp_param = exp_param.to(self.data_device)
        self.eyes_pose = eyes_pose.to(self.data_device)
        self.eyelids = eyelids.to(self.data_device)
        self.jaw_pose = jaw_pose.to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]



#  smirkcamera class
class Camera_smirk(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, head_mask, mouth_mask,
                 exp_param, pose_param, eyelids, jaw_pose,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera_smirk, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.head_mask = head_mask.to(self.data_device)
        self.mouth_mask = mouth_mask.to(self.data_device)
        self.exp_param = exp_param.to(self.data_device)
        # self.eyes_pose = eyes_pose.to(self.data_device)
        self.pose_param = pose_param.to(self.data_device)
        self.eyelids = eyelids.to(self.data_device)
        self.jaw_pose = jaw_pose.to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
class Camera_dad(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, head_mask, mouth_mask,
                 exp_param, eyes_pose, eyelids, jaw_pose,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera_dad, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.head_mask = head_mask.to(self.data_device)
        self.mouth_mask = mouth_mask.to(self.data_device)
        self.exp_param = exp_param.to(self.data_device)
        self.eyes_pose = eyes_pose.to(self.data_device)
        self.eyelids = eyelids.to(self.data_device)
        self.jaw_pose = jaw_pose.to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]



#  smirkcamera class
class Camera_smirk(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, head_mask, mouth_mask,
                 exp_param, pose_param, eyelids, jaw_pose,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera_smirk, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.head_mask = head_mask.to(self.data_device)
        self.mouth_mask = mouth_mask.to(self.data_device)
        self.exp_param = exp_param.to(self.data_device)
        # self.eyes_pose = eyes_pose.to(self.data_device)
        self.pose_param = pose_param.to(self.data_device)
        self.eyelids = eyelids.to(self.data_device)
        self.jaw_pose = jaw_pose.to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        def load_images(self):
            pass
       
class Camera_smirk_light(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image_path, head_mask_path, 
                 mouth_mask_path, alpha_path, exp_param, pose_param, eyelids, jaw_pose,
                 image_name, uid, data_device, bg_image,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        super(Camera_smirk_light, self).__init__()
        
        # 基本参数
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        
        # 图像路径
        self.image_path = image_path
        self.head_mask_path = head_mask_path
        self.mouth_mask_path = mouth_mask_path
        self.alpha_path = alpha_path
        self.bg_image = bg_image
        
        # 设备设置
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
            
        # 变形参数
        self.exp_param = exp_param.to(self.data_device)
        self.pose_param = pose_param.to(self.data_device)
        self.eyelids = eyelids.to(self.data_device)
        self.jaw_pose = jaw_pose.to(self.data_device)
        
        # 相机参数
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale
        
        # 设置图像尺寸（从一个临时加载的图像获取）
        with Image.open(self.image_path) as img:
            self.image_width = img.size[0]
            self.image_height = img.size[1]
        
        # 计算变换矩阵
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
    def load_images(self):
        """按需加载图像数据"""
        if not hasattr(self, 'original_image'):
            # 加载原始图像
            image = PILtoTensor(Image.open(self.image_path))
            gt_image = image[:3, ...]
            
            # 加载alpha
            alpha = PILtoTensor(Image.open(self.alpha_path))
            
            # 加载head mask
            head_mask = PILtoTensor(Image.open(self.head_mask_path))
            
            # 加载mouth mask
            mouth_mask = PILtoTensor(Image.open(self.mouth_mask_path))
            
            # 应用mask
            gt_image = gt_image * alpha + self.bg_image * (1 - alpha)
            gt_image = gt_image * head_mask + self.bg_image * (1 - head_mask)
            
            # 存储到设备上
            self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
            self.head_mask = head_mask.to(self.data_device)
            self.mouth_mask = mouth_mask.to(self.data_device)

# emoca camera class
class Camera_emoca(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, head_mask, mouth_mask,
                 exp_param, pose_param, detail,
                 image_name, uid,cam_params = None, shape_param=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera_emoca, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.head_mask = head_mask.to(self.data_device)
        self.mouth_mask = mouth_mask.to(self.data_device)
        # self.shape_param = shape_param.to(self.data_device)
        self.exp_param = exp_param.to(self.data_device)
        # self.cam_param = cam_params.to(self.data_device)
        
        # # old
        # self.eyes_pose = eyes_pose.to(self.data_device)
        # self.eyelids = eyelids.to(self.data_device)
        # self.jaw_pose = jaw_pose.to(self.data_device)
        
        # emoca
        self.pose_param = pose_param.to(self.data_device)
        self.detail = detail.to(self.data_device)
        

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # self.world_view_transform[3,1]+=0.1
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

