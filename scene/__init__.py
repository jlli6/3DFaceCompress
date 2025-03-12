import os, sys
import random
import json
from PIL import Image
import torch
import math
import numpy as np
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from scene.gaussian_model_sq import GaussianModelSQ
from scene.cameras import Camera,Camera_smirk,Camera_emoca,Camera_smirk_light
from arguments import ModelParams
from utils.general_utils import PILtoTensor
from utils.graphics_utils import focal2fov
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, axis_angle_to_matrix,euler_angles_to_matrix,matrix_to_axis_angle

def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)

def quantize(param, max_val, min_val, bits):
    """
    对参数进行量化
    :param param: 原始参数 (Tensor)
    :param max_val: 每个维度的最大值 (Tensor)
    :param min_val: 每个维度的最小值 (Tensor)
    :param bits: 量化的位数 (8 或 10)
    :return: 量化后的 Tensor
    """
    levels = 2 ** bits - 1  # 量化级数，8-bit 为 255，10-bit 为 1023
    # 防止除零
    range_val = max_val - min_val
    range_val[range_val == 0] = 1e-6  # 避免出现除零

    # 量化公式
    quantized = ((param - min_val) / range_val * levels).round().clamp(0, levels)

    return quantized.type(torch.uint8 if bits == 8 else torch.uint16)

def dequantize(quantized, max_val, min_val, bits):
    levels = 2 ** bits - 1
    quantized = np.clip(quantized, 0, levels)
    return quantized / levels * (max_val - min_val) + min_val

class Scene_mica:
    def __init__(self, datadir, mica_datadir, train_type, white_background, device):
        ## train_type: 0 for train, 1 for test, 2 for eval
        frame_delta = 1 # default mica-tracking starts from the second frame
        images_folder = os.path.join(datadir, "imgs")
        parsing_folder = os.path.join(datadir, "parsing")
        alpha_folder = os.path.join(datadir, "alpha")
        
        self.bg_image = torch.zeros((3, 512, 512))
        if white_background:
            self.bg_image[:, :, :] = 1
        else:
            self.bg_image[1, :, :] = 1

        mica_ckpt_dir = os.path.join(mica_datadir, 'checkpoint')
        # mica_ckpt_dir = os.path.join(mica_datadir, 'checkpoint_audio_exp')
        # mica_ckpt_dir = os.path.join(mica_datadir, 'checkpoint_exp_only')
        # mica_ckpt_dir = "/home/xylem/IBC24/metrical-tracker/output/id2_test/checkpoint_half"
        self.N_frames = len(os.listdir(mica_ckpt_dir))
        self.cameras = []
        test_num = 500 #1400
        # test_num = 50
        eval_num = 50
        max_train_num = 200
        
        train_num = min(max_train_num, self.N_frames - test_num)
        ckpt_path = os.path.join(mica_ckpt_dir, '00000.frame')# 00000.frame
        payload = torch.load(ckpt_path)
        flame_params = payload['flame']
        self.shape_param = torch.as_tensor(flame_params['shape'])
        orig_w, orig_h = payload['img_size']
        K = payload['opencv']['K'][0]
        fl_x = K[0, 0]
        fl_y = K[1, 1]
        FovY = focal2fov(fl_y, orig_h)
        FovX = focal2fov(fl_x, orig_w)
        if train_type == 0:
            range_down = 0  # 从500帧开始，有的人前面的数据不好
            range_up = train_num + range_down
        if train_type == 1:
            range_down = self.N_frames - test_num
            range_up = self.N_frames
            # range_down = 10
            # range_up = 11
        if train_type == 2:
            range_down = self.N_frames - eval_num
            range_up = self.N_frames
        if train_type == 3:
            range_down = 0
            range_up = self.N_frames
            

        for frame_id in tqdm(range(range_down, range_up)):
            image_name_mica = str(frame_id).zfill(5) # obey mica tracking
            image_name_ori = str(frame_id+frame_delta).zfill(5)
            ckpt_path = os.path.join(mica_ckpt_dir, image_name_mica+'.frame')
            payload = torch.load(ckpt_path)
            
            flame_params = payload['flame']
            
            # ########### old training code
            exp_param = torch.as_tensor(flame_params['exp'])
            eyes_pose = torch.as_tensor(flame_params['eyes'])
            eyelids = torch.as_tensor(flame_params['eyelids'])
            jaw_pose = torch.as_tensor(flame_params['jaw'])
            
            # ########## emoca training code
            # exp_param = torch.as_tensor(flame_params['exp'])
            # pose_param = torch.as_tensor(flame_params['pose'])
            # detail = torch.as_tensor(flame_params['detail'])
            
            # ############ smirk training code
            # exp_param = torch.as_tensor(flame_params['exp'])
            # pose_param = torch.as_tensor(flame_params['pose'])
            # # eyes_pose = torch.as_tensor(flame_params['eyes'])
            # eyelids = torch.as_tensor(flame_params['eyelids'])
            # jaw_pose = torch.as_tensor(flame_params['jaw'])

            oepncv = payload['opencv']
            w2cR = oepncv['R'][0]
            w2cT = oepncv['t'][0] 
            # scale = oepncv['scale'][0]
            R = np.transpose(w2cR) # R is stored transposed due to 'glm' in CUDA code
            T = w2cT
            # translate =  oepncv['t'][0] 

            # 有的是jpg，有的是png
            for ext in ['.png', '.jpg']:
                image_path = os.path.join(images_folder, image_name_ori + ext)
                if os.path.exists(image_path):
                    break  # 找到匹配的文件后跳出循环
            else:
                raise FileNotFoundError(f"No image found for {image_name_ori} with .png or .jpg extension in {images_folder}")
            # image_path = os.path.join(images_folder, image_name_ori+'.png')
            
            image = Image.open(image_path)
            resized_image_rgb = PILtoTensor(image)
            gt_image = resized_image_rgb[:3, ...]
            
            # alpha
            alpha_path = os.path.join(alpha_folder, image_name_ori+'.jpg')
            alpha = Image.open(alpha_path)
            alpha = PILtoTensor(alpha)

            # # if add head mask
            head_mask_path = os.path.join(parsing_folder, image_name_ori+'_neckhead.png')
            head_mask = Image.open(head_mask_path)
            head_mask = PILtoTensor(head_mask)
            gt_image = gt_image * alpha + self.bg_image * (1 - alpha)
            gt_image = gt_image * head_mask + self.bg_image * (1 - head_mask)

            # mouth mask
            mouth_mask_path = os.path.join(parsing_folder, image_name_ori+'_mouth.png') # jpg
            mouth_mask = Image.open(mouth_mask_path)
            mouth_mask = PILtoTensor(mouth_mask)
            # if frame_id % 100 == 0:
            #     print(f" mouth 张量的形状: {mouth_mask.size()}")
            #     tensor_memory_size = mouth_mask.element_size() * mouth_mask.numel()
            #     print(" mouth element_size: ", mouth_mask.element_size())
            #     print(" mouth numel: ", mouth_mask.numel())

            #     # 输出张量的大小，单位为字节
            #     print(f"mouth 张量占用内存大小: {tensor_memory_size} 字节 ({tensor_memory_size / 1024:.2f} KB)")
                
            #     # alhpa 空间占用
            #     print(f"alpha 张量的形状: {alpha.size()}")
            #     tensor_memory_size = alpha.element_size() * alpha.numel()
            #     print("alpha element_size: ", alpha.element_size())
            #     print("alpha numel: ", alpha.numel())
            #     print(f"alpha 张量占用内存大小: {tensor_memory_size} 字节 ({tensor_memory_size / 1024:.2f} KB)")
                
            #     # head_mask 空间占用
                
            #     print(f"head_mask 张量的形状: {head_mask.size()}")
            #     tensor_memory_size = head_mask.element_size() * head_mask.numel()
            #     print("head_mask element_size: ", head_mask.element_size())
            #     print("head_mask numel: ", head_mask.numel())
            #     print(f"head_mask 张量占用内存大小: {tensor_memory_size} 字节 ({tensor_memory_size / 1024:.2f} KB)")
                
            #     # GT
            #     print(f"gt_image 张量的形状: {gt_image.size()}")
            #     tensor_memory_size = gt_image.element_size() * gt_image.numel()
            #     print("gt_image element_size: ", gt_image.element_size())
            #     print("gt_image numel: ", gt_image.numel())
            #     print(f"gt_image 张量占用内存大小: {tensor_memory_size} 字节 ({tensor_memory_size / 1024:.2f} KB)")
            
            
            # old camera
            camera_indiv = Camera(colmap_id=frame_id, R=R, T=T,
                                FoVx=FovX, FoVy=FovY, 
                                image=gt_image, head_mask=head_mask, mouth_mask=mouth_mask,
                                exp_param=exp_param, eyes_pose=eyes_pose, eyelids=eyelids, jaw_pose=jaw_pose,
                                image_name=image_name_mica, uid=frame_id, data_device=device)
            
            # # emoca camera
            # camera_indiv = Camera(colmap_id=frame_id, R=R, T=T, 
            #                     FoVx=FovX, FoVy=FovY, 
            #                     image=gt_image, head_mask=head_mask, mouth_mask=mouth_mask,
            #                     exp_param=exp_param, pose_param=pose_param, detail=detail,
            #                     image_name=image_name_mica, uid=frame_id, data_device=device)
            
            # # smirk camera
            # camera_indiv = Camera(colmap_id=frame_id, R=R, T=T, 
            #                     FoVx=FovX, FoVy=FovY, 
            #                     image=gt_image, head_mask=head_mask, mouth_mask=mouth_mask,
            #                     exp_param=exp_param, pose_param=pose_param, eyelids=eyelids, jaw_pose=jaw_pose,
            #                     image_name=image_name_mica, uid=frame_id, data_device=device)
            self.cameras.append(camera_indiv)
    
    def getCameras(self):
        return self.cameras
    
class Scene_mica_dad:
    def __init__(self, datadir, mica_datadir, train_type, white_background, device):
        ## train_type: 0 for train, 1 for test, 2 for eval
        frame_delta = 1 # default mica-tracking starts from the second frame
        images_folder = os.path.join(datadir, "imgs")
        parsing_folder = os.path.join(datadir, "parsing")
        alpha_folder = os.path.join(datadir, "alpha")
        
        self.bg_image = torch.zeros((3, 512, 512))
        if white_background:
            self.bg_image[:, :, :] = 1
        else:
            self.bg_image[1, :, :] = 1

        mica_ckpt_dir = os.path.join(mica_datadir, 'checkpoint')
        # mica_ckpt_dir = os.path.join(mica_datadir, 'checkpoint_audio_exp')
        # mica_ckpt_dir = os.path.join(mica_datadir, 'checkpoint_exp_only')
        # mica_ckpt_dir = "/home/xylem/IBC24/metrical-tracker/output/id2_test/checkpoint_half"
        self.N_frames = len(os.listdir(mica_ckpt_dir))
        self.cameras = []
        test_num = 500 #1400
        # test_num = 50
        eval_num = 50
        max_train_num = 2000
        
        train_num = min(max_train_num, self.N_frames - test_num)
        ckpt_path = os.path.join(mica_ckpt_dir, '00000.frame')# 00000.frame
        payload = torch.load(ckpt_path)
        flame_params = payload['flame']
        self.shape_param = torch.as_tensor(flame_params['shape'])
        orig_w, orig_h = payload['img_size']
        K = payload['opencv']['K'][0]
        fl_x = K[0, 0]
        fl_y = K[1, 1]
        FovY = focal2fov(fl_y, orig_h)
        FovX = focal2fov(fl_x, orig_w)
        if train_type == 0:
            range_down = 0  # 从500帧开始，有的人前面的数据不好
            range_up = train_num + range_down
        if train_type == 1:
            range_down = self.N_frames - test_num
            range_up = self.N_frames
            # range_down = 10
            # range_up = 11
        if train_type == 2:
            range_down = self.N_frames - eval_num
            range_up = self.N_frames
        if train_type == 3:
            range_down = 0
            range_up = self.N_frames
            

        for frame_id in tqdm(range(range_down, range_up)):
            image_name_mica = str(frame_id).zfill(5) # obey mica tracking
            image_name_ori = str(frame_id+frame_delta).zfill(5)
            ckpt_path = os.path.join(mica_ckpt_dir, image_name_mica+'.frame')
            payload = torch.load(ckpt_path)
            
            flame_params = payload['flame']
            
            # ########### old training code
            exp_param = torch.as_tensor(flame_params['exp'])
            eyes_pose = torch.as_tensor(flame_params['eyes'])
            eyelids = torch.as_tensor(flame_params['eyelids'])
            jaw_pose = torch.as_tensor(flame_params['jaw'])
            
            # ########## emoca training code
            # exp_param = torch.as_tensor(flame_params['exp'])
            # pose_param = torch.as_tensor(flame_params['pose'])
            # detail = torch.as_tensor(flame_params['detail'])
            
            # ############ smirk training code
            # exp_param = torch.as_tensor(flame_params['exp'])
            # pose_param = torch.as_tensor(flame_params['pose'])
            # # eyes_pose = torch.as_tensor(flame_params['eyes'])
            # eyelids = torch.as_tensor(flame_params['eyelids'])
            # jaw_pose = torch.as_tensor(flame_params['jaw'])

            oepncv = payload['opencv']
            w2cR = oepncv['R'][0]
            w2cT = oepncv['t'][0] 
            w2T_scale = oepncv['scale'][0]
            R = np.transpose(w2cR) # R is stored transposed due to 'glm' in CUDA code
            T = w2cT

            # 有的是jpg，有的是png
            for ext in ['.png', '.jpg']:
                image_path = os.path.join(images_folder, image_name_ori + ext)
                if os.path.exists(image_path):
                    break  # 找到匹配的文件后跳出循环
            else:
                raise FileNotFoundError(f"No image found for {image_name_ori} with .png or .jpg extension in {images_folder}")
            # image_path = os.path.join(images_folder, image_name_ori+'.png')
            
            image = Image.open(image_path)
            resized_image_rgb = PILtoTensor(image)
            gt_image = resized_image_rgb[:3, ...]
            
            # alpha
            alpha_path = os.path.join(alpha_folder, image_name_ori+'.jpg')
            alpha = Image.open(alpha_path)
            alpha = PILtoTensor(alpha)

            # # if add head mask
            head_mask_path = os.path.join(parsing_folder, image_name_ori+'_neckhead.png')
            head_mask = Image.open(head_mask_path)
            head_mask = PILtoTensor(head_mask)
            gt_image = gt_image * alpha + self.bg_image * (1 - alpha)
            gt_image = gt_image * head_mask + self.bg_image * (1 - head_mask)

            # mouth mask
            mouth_mask_path = os.path.join(parsing_folder, image_name_ori+'_mouth.png') # jpg
            mouth_mask = Image.open(mouth_mask_path)
            mouth_mask = PILtoTensor(mouth_mask)
            # if frame_id % 100 == 0:
            #     print(f" mouth 张量的形状: {mouth_mask.size()}")
            #     tensor_memory_size = mouth_mask.element_size() * mouth_mask.numel()
            #     print(" mouth element_size: ", mouth_mask.element_size())
            #     print(" mouth numel: ", mouth_mask.numel())

            #     # 输出张量的大小，单位为字节
            #     print(f"mouth 张量占用内存大小: {tensor_memory_size} 字节 ({tensor_memory_size / 1024:.2f} KB)")
                
            #     # alhpa 空间占用
            #     print(f"alpha 张量的形状: {alpha.size()}")
            #     tensor_memory_size = alpha.element_size() * alpha.numel()
            #     print("alpha element_size: ", alpha.element_size())
            #     print("alpha numel: ", alpha.numel())
            #     print(f"alpha 张量占用内存大小: {tensor_memory_size} 字节 ({tensor_memory_size / 1024:.2f} KB)")
                
            #     # head_mask 空间占用
                
            #     print(f"head_mask 张量的形状: {head_mask.size()}")
            #     tensor_memory_size = head_mask.element_size() * head_mask.numel()
            #     print("head_mask element_size: ", head_mask.element_size())
            #     print("head_mask numel: ", head_mask.numel())
            #     print(f"head_mask 张量占用内存大小: {tensor_memory_size} 字节 ({tensor_memory_size / 1024:.2f} KB)")
                
            #     # GT
            #     print(f"gt_image 张量的形状: {gt_image.size()}")
            #     tensor_memory_size = gt_image.element_size() * gt_image.numel()
            #     print("gt_image element_size: ", gt_image.element_size())
            #     print("gt_image numel: ", gt_image.numel())
            #     print(f"gt_image 张量占用内存大小: {tensor_memory_size} 字节 ({tensor_memory_size / 1024:.2f} KB)")
            
            
            # old camera
            camera_indiv = Camera(colmap_id=frame_id, R=R, T=T, 
                                FoVx=FovX, FoVy=FovY, 
                                image=gt_image, head_mask=head_mask, mouth_mask=mouth_mask,
                                exp_param=exp_param, eyes_pose=eyes_pose, eyelids=eyelids, jaw_pose=jaw_pose,
                                image_name=image_name_mica, uid=frame_id, data_device=device)
            
            # # emoca camera
            # camera_indiv = Camera(colmap_id=frame_id, R=R, T=T, 
            #                     FoVx=FovX, FoVy=FovY, 
            #                     image=gt_image, head_mask=head_mask, mouth_mask=mouth_mask,
            #                     exp_param=exp_param, pose_param=pose_param, detail=detail,
            #                     image_name=image_name_mica, uid=frame_id, data_device=device)
            
            # # smirk camera
            # camera_indiv = Camera(colmap_id=frame_id, R=R, T=T, 
            #                     FoVx=FovX, FoVy=FovY, 
            #                     image=gt_image, head_mask=head_mask, mouth_mask=mouth_mask,
            #                     exp_param=exp_param, pose_param=pose_param, eyelids=eyelids, jaw_pose=jaw_pose,
            #                     image_name=image_name_mica, uid=frame_id, data_device=device)
            self.cameras.append(camera_indiv)
    
    def getCameras(self):
        return self.cameras
    
class Scene_mica_emoca:
    def __init__(self, datadir, mica_datadir, train_type, white_background, device, id_name = None, quan_bits = 0):
        ## train_type: 0 for train, 1 for test, 2 for eval
        frame_delta = 1 # default mica-tracking starts from the second frame
        images_folder = os.path.join(datadir, "imgs")
        parsing_folder = os.path.join(datadir, "parsing")
        alpha_folder = os.path.join(datadir, "alpha")
        
        self.bg_image = torch.zeros((3, 512, 512))
        if white_background:
            self.bg_image[:, :, :] = 1
        else:
            self.bg_image[1, :, :] = 1

        mica_ckpt_dir = os.path.join(mica_datadir, 'checkpoint_offset')
        # mica_ckpt_dir = "/home/xylem/IBC24/metrical-tracker/output/id2_test/checkpoint_half"
        self.N_frames = len(os.listdir(mica_ckpt_dir))
        self.cameras = []
        test_num = 500 #1400, 500
        # test_num = 50
        eval_num = 50
        max_train_num = 2000
        train_num = min(max_train_num, self.N_frames - test_num)
        ckpt_path = os.path.join(mica_ckpt_dir, '00000.frame')# 00000.frame
        payload = torch.load(ckpt_path)
        flame_params = payload['flame']
        self.shape_param = torch.as_tensor(flame_params['shape'])
        orig_w, orig_h = payload['img_size']
        K = payload['opencv']['K'][0]
        fl_x = K[0, 0]
        fl_y = K[1, 1]
        FovY = focal2fov(fl_y, orig_h)
        FovX = focal2fov(fl_x, orig_w)
        if train_type == 0:
            range_down = 300
            range_up = train_num +range_down
        if train_type == 1:
            range_down = self.N_frames - test_num
            range_up = self.N_frames
            # range_down = 1
            # range_up = 501
        if train_type == 2:
            range_down = self.N_frames - eval_num
            range_up = self.N_frames
            # range_up = range_down +1
            
        if train_type == 3:
            range_down = 0
            # range_up = self.N_frames
            range_up = 1
            

        if  quan_bits != 0:
            max_flame_path = os.path.join(mica_datadir, f'{id_name}_max.frame')
            min_flame_path = os.path.join(mica_datadir, f'{id_name}_min.frame') 
            max_flame = torch.load(max_flame_path)  # 形状和 flame_params['exp'], 'pose' 相同
            min_flame = torch.load(min_flame_path)  # 形状和 flame_params['exp'], 'pose' 相同
            
            max_opencv_t = np.array([[0.2,0.3,5]],dtype=np.float32)
            min_opencv_t = np.array([[-0.2,-0.3,0.5]],dtype=np.float32)
            max_opencv_t_tensor = torch.as_tensor(max_opencv_t)
            min_opencv_t_tensor = torch.as_tensor(min_opencv_t)
            
        for frame_id in tqdm(range(range_down, range_up)):
            image_name_mica = str(frame_id).zfill(5) # obey mica tracking
            image_name_ori = str(frame_id+frame_delta).zfill(5)
            ckpt_path = os.path.join(mica_ckpt_dir, image_name_mica+'.frame')
            payload = torch.load(ckpt_path)
            
            flame_params = payload['flame']
            
            # if use_quan:
            
                
            ########## emoca training code
            exp_param = torch.as_tensor(flame_params['exp'])
            pose_param = torch.as_tensor(flame_params['pose'])
            detail = torch.as_tensor(flame_params['detail'])
            # shape_param = torch.as_tensor(flame_params['shape'])
            # camera_params = torch.as_tensor(np.array([[1.0019892e+01, 1.4389530e-03, 3.7328370e-02]])
            oepncv = payload['opencv']
            w2cR = oepncv['R'][0]
            w2cT = oepncv['t'][0] 
                
            
            #对表情参数，pose参数，opencv['t'],opencv['R']进行量化,量化选择8bit 或者10bit, 0表示不量化
            if quan_bits != 0:
                detail = torch.as_tensor(flame_params['detail'])
                exp_param = torch.as_tensor(flame_params['exp'])
                pose_param = torch.as_tensor(flame_params['pose'])
                
                # 表情经过量化，反量化
                exp_param_8bit = quantize(exp_param, max_flame['flame']['exp'], min_flame['flame']['exp'], bits=quan_bits)
                exp_param_8bit_rec = dequantize(exp_param_8bit, max_flame['flame']['exp'], min_flame['flame']['exp'], bits=quan_bits)
                exp_param = exp_param_8bit_rec
                
                # pose经过量化，反量化
                pose_param_8bit = quantize(pose_param, max_flame['flame']['pose'], min_flame['flame']['pose'], bits=quan_bits)
                pose_param_8bit_rec = dequantize(pose_param_8bit, max_flame['flame']['pose'], min_flame['flame']['pose'], bits=quan_bits)
                pose_param = pose_param_8bit_rec
                
                # opencv['t']经过量化，反量化
                oepncv_t_8bit = quantize(torch.as_tensor(oepncv['t'][0]), max_opencv_t_tensor, min_opencv_t_tensor, bits=quan_bits)
                oepncv_t_8bit_rec = dequantize(oepncv_t_8bit, max_opencv_t_tensor, min_opencv_t_tensor, bits=quan_bits)
                
                oepncv['t'][0] = oepncv_t_8bit_rec
                
                # opencv['R']先从矩阵变成轴角，轴角经过量化，反量化，再变成矩阵
                oepncv_R_matrix = torch.as_tensor(oepncv['R'][0])
                oepncv_R_axis_angle = matrix_to_axis_angle(oepncv_R_matrix)
                oepncv_R_axis_angle_8bit = quantize(oepncv_R_axis_angle, torch.tensor(np.pi), torch.tensor(-np.pi), bits=quan_bits)
                oepncv_R_axis_angle_8bit_rec = dequantize(oepncv_R_axis_angle_8bit, torch.tensor(np.pi), torch.tensor(-np.pi), bits=quan_bits)
                oepncv_R_matrix_rec = axis_angle_to_matrix(oepncv_R_axis_angle_8bit_rec)    
                oepncv['R'][0] = oepncv_R_matrix_rec
                
                w2cR = oepncv['R'][0]
                w2cT = oepncv['t'][0] 
                
            R = np.transpose(w2cR) # R is stored transposed due to 'glm' in CUDA code
            T = w2cT

            # 有的是jpg，有的是png
            for ext in ['.png', '.jpg']:
                image_path = os.path.join(images_folder, image_name_ori + ext)
                if os.path.exists(image_path):
                    break  # 找到匹配的文件后跳出循环
            else:
                raise FileNotFoundError(f"No image found for {image_name_ori} with .png or .jpg extension in {images_folder}")
            
            # image_path = os.path.join(images_folder, image_name_ori+'.png')
            image = Image.open(image_path)
            resized_image_rgb = PILtoTensor(image)
            gt_image = resized_image_rgb[:3, ...]
            
            # alpha
            alpha_path = os.path.join(alpha_folder, image_name_ori+'.jpg')
            alpha = Image.open(alpha_path)
            alpha = PILtoTensor(alpha)

            # # if add head mask
            head_mask_path = os.path.join(parsing_folder, image_name_ori+'_neckhead.png')
            head_mask = Image.open(head_mask_path)
            head_mask = PILtoTensor(head_mask)
            gt_image = gt_image * alpha + self.bg_image * (1 - alpha)
            gt_image = gt_image * head_mask + self.bg_image * (1 - head_mask)

            # mouth mask
            mouth_mask_path = os.path.join(parsing_folder, image_name_ori+'_mouth.png') # jpg
            mouth_mask = Image.open(mouth_mask_path)
            mouth_mask = PILtoTensor(mouth_mask)
           
            
            
            # # old camera
            # camera_indiv = Camera(colmap_id=frame_id, R=R, T=T, 
            #                     FoVx=FovX, FoVy=FovY, 
            #                     image=gt_image, head_mask=head_mask, mouth_mask=mouth_mask,
            #                     exp_param=exp_param, eyes_pose=eyes_pose, eyelids=eyelids, jaw_pose=jaw_pose,
            #                     image_name=image_name_mica, uid=frame_id, data_device=device)
            
            # emoca camera
            camera_indiv = Camera_emoca(colmap_id=frame_id, R=R, T=T, 
                                FoVx=FovX, FoVy=FovY, 
                                image=gt_image, head_mask=head_mask, mouth_mask=mouth_mask,
                                exp_param=exp_param, pose_param=pose_param, detail=detail,
                                image_name=image_name_mica, uid=frame_id,  data_device=device)
            
            # # smirk camera
            # camera_indiv = Camera(colmap_id=frame_id, R=R, T=T, 
            #                     FoVx=FovX, FoVy=FovY, 
            #                     image=gt_image, head_mask=head_mask, mouth_mask=mouth_mask,
            #                     exp_param=exp_param, pose_param=pose_param, eyelids=eyelids, jaw_pose=jaw_pose,
            #                     image_name=image_name_mica, uid=frame_id, data_device=device)
            self.cameras.append(camera_indiv)
    
    def getCameras(self):
        return self.cameras


class Scene_mica_smirk:
    def __init__(self, datadir, mica_datadir, train_type, white_background, device, id_name = None,  quan_bits = 0):
        ## train_type: 0 for train, 1 for test, 2 for eval
        frame_delta = 1 # default mica-tracking starts from the second frame
        images_folder = os.path.join(datadir, "imgs")
        parsing_folder = os.path.join(datadir, "parsing")
        alpha_folder = os.path.join(datadir, "alpha")
        
        self.bg_image = torch.zeros((3, 512, 512))
        if  white_background:
            self.bg_image[:, :, :] = 1
        else:
            self.bg_image[1, :, :] = 1

        mica_ckpt_dir = os.path.join(mica_datadir, 'checkpoint_offset')
        # mica_ckpt_dir = mica_datadir
        # mica_ckpt_dir = "/home/xylem/IBC24/metrical-tracker/output/id2_test/checkpoint_half"
        self.N_frames = len(os.listdir(mica_ckpt_dir))
        self.cameras = []
        test_num = 500 #1400
        # test_num = 50
        eval_num = 50
        max_train_num = 2000
        train_num = min(max_train_num, self.N_frames - test_num)
        ckpt_path = os.path.join(mica_ckpt_dir, '00000.frame')# 00000.frame
        payload = torch.load(ckpt_path)
        flame_params = payload['flame']
        self.shape_param = torch.as_tensor(flame_params['shape'])
        orig_w, orig_h = payload['img_size']
        K = payload['opencv']['K'][0]
        # K = np.array([[[2.3742700e+03, 0.0000000e+00, 2.5805597e+02],
        #                 [0.0000000e+00, 2.3742700e+03, 2.5243530e+02],
        #                 [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]]])[0]
        fl_x = K[0, 0]
        fl_y = K[1, 1]
        FovY = focal2fov(fl_y, orig_h)
        FovX = focal2fov(fl_x, orig_w)
        if train_type == 0:
            range_down = 0  ## 从300帧开始，有的人前面的数据不好
            range_up = train_num + range_down
        if train_type == 1:
            range_down = self.N_frames - test_num
            range_up = self.N_frames
            # range_down = 10
            # range_up = 11
        if train_type == 2:
            range_down = self.N_frames - eval_num
            range_up = self.N_frames
            
        if train_type == 3:
            range_down = 0
            range_up = 1000
        
        if  quan_bits != 0:
            print("using quantization", quan_bits,"bits, current id_name: ", id_name)
            max_flame_path = os.path.join(mica_datadir, f'{id_name}_max.frame')
            min_flame_path = os.path.join(mica_datadir, f'{id_name}_min.frame') 
            max_flame = torch.load(max_flame_path)  # 形状和 flame_params['exp'], 'pose' 相同
            min_flame = torch.load(min_flame_path)  # 形状和 flame_params['exp'], 'pose' 相同
            
            max_opencv_t = np.array([[0.2,0.3,5]],dtype=np.float32)
            min_opencv_t = np.array([[-0.2,-0.3,0.5]],dtype=np.float32)
            max_opencv_t_tensor = torch.as_tensor(max_opencv_t)
            min_opencv_t_tensor = torch.as_tensor(min_opencv_t)
            
        for frame_id in tqdm(range(range_down, range_up)):
            image_name_mica = str(frame_id).zfill(5) # obey mica tracking
            image_name_ori = str(frame_id+frame_delta).zfill(5)
            ckpt_path = os.path.join(mica_ckpt_dir, image_name_mica+'.frame')
            payload = torch.load(ckpt_path) 
            
            flame_params = payload['flame']
            
            ############ smirk training code
            exp_param = torch.as_tensor(flame_params['exp'])
            pose_param = torch.as_tensor(flame_params['pose'])
            # eyes_pose = torch.as_tensor(flame_params['eyes'])
            eyelids = torch.as_tensor(flame_params['eyelids'])
            jaw_pose = torch.as_tensor(flame_params['jaw'])

            oepncv = payload['opencv']
            
            w2cR = oepncv['R'][0] # mica opencv
            w2cT = oepncv['t'][0]
            # w2cR = np.array([[1,0,0],[0,-1,0],[0,0,-1]]) # smirk cam
            
    
            # # 以下代码对smirk的pose输出做修改，使其适应flashavatar数据集输入
            # cam  = payload['camera']['cam'].copy()
            # w2cT = cam[0].copy()
            # w2cT[0] = cam[0][1]
            # w2cT[1] = -cam[0][2]
            # # w2cT[2] = -0.0661* cam[0][0] + 1.973
            # w2cT[2] = (1/cam[0][0])*9.629
            
            # zero_rotation_angle = np.array([np.pi, 0, 0])  # 绕 z 轴旋转 180°

            # zero_rotation = to_tensor(zero_rotation_angle)
            # zero_rotation_matrix = axis_angle_to_matrix(zero_rotation).numpy()
            
            # # smirk_rotation_angle = pose_param.clone()
            
            # # smirk_rotation_matrix = axis_angle_to_matrix(smirk_rotation_angle).numpy()
            # w2cR = zero_rotation_matrix
            
             #对表情参数，pose参数，opencv['t'],opencv['R']进行量化,量化选择8bit 或者10bit, 0表示不量化
            if quan_bits != 0:
                
                exp_param = torch.as_tensor(flame_params['exp'])
                pose_param = torch.as_tensor(flame_params['pose'])
                jaw_pose= torch.as_tensor(flame_params['jaw'])
                eyelids = torch.as_tensor(flame_params['eyelids'])
                
                # 表情经过量化，反量化
                exp_param_8bit = quantize(exp_param, max_flame['flame']['exp'], min_flame['flame']['exp'], bits=quan_bits)
                exp_param_8bit_rec = dequantize(exp_param_8bit, max_flame['flame']['exp'], min_flame['flame']['exp'], bits=quan_bits)
                exp_param = exp_param_8bit_rec
                
                # pose经过量化，反量化
                pose_param_8bit = quantize(pose_param, max_flame['flame']['pose'], min_flame['flame']['pose'], bits=quan_bits)
                pose_param_8bit_rec = dequantize(pose_param_8bit, max_flame['flame']['pose'], min_flame['flame']['pose'], bits=quan_bits)
                pose_param = pose_param_8bit_rec
                
                # jaw 经过量化反量化
                jaw_pose_8bit = quantize(jaw_pose, max_flame['flame']['jaw'], min_flame['flame']['jaw'], bits=quan_bits)
                jaw_pose_8bit_rec = dequantize(jaw_pose_8bit, max_flame['flame']['jaw'], min_flame['flame']['jaw'], bits=quan_bits)
                jaw_pose = jaw_pose_8bit_rec
                
                # eyelids 经过量化反量化，其最大值和最小值分别是[0,0],[1,1]
                eyelids_8bit = quantize(eyelids, torch.tensor([1,1]), torch.tensor([0,0]), bits=quan_bits)
                eyelids_8bit_rec = dequantize(eyelids_8bit, torch.tensor([1,1]), torch.tensor([0,0]), bits=quan_bits)
                eyelids = eyelids_8bit_rec
                
                # opencv['t']经过量化，反量化
                oepncv_t_8bit = quantize(torch.as_tensor(oepncv['t'][0]), max_opencv_t_tensor, min_opencv_t_tensor, bits=quan_bits)
                oepncv_t_8bit_rec = dequantize(oepncv_t_8bit, max_opencv_t_tensor, min_opencv_t_tensor, bits=quan_bits)
                
                oepncv['t'][0] = oepncv_t_8bit_rec
                
                # opencv['R']先从矩阵变成轴角，轴角经过量化，反量化，再变成矩阵
                oepncv_R_matrix = torch.as_tensor(oepncv['R'][0])
                oepncv_R_axis_angle = matrix_to_axis_angle(oepncv_R_matrix)
                oepncv_R_axis_angle_8bit = quantize(oepncv_R_axis_angle, torch.tensor(np.pi), torch.tensor(-np.pi), bits=quan_bits)
                oepncv_R_axis_angle_8bit_rec = dequantize(oepncv_R_axis_angle_8bit, torch.tensor(np.pi), torch.tensor(-np.pi), bits=quan_bits)
                oepncv_R_matrix_rec = axis_angle_to_matrix(oepncv_R_axis_angle_8bit_rec)    
                oepncv['R'][0] = oepncv_R_matrix_rec
                
                w2cR = oepncv['R'][0]
                w2cT = oepncv['t'][0] 
                
            R = np.transpose(w2cR) # R is stored transposed due to 'glm' in CUDA code
            T = w2cT

            # 有的是jpg，有的是png
            for ext in ['.png', '.jpg']:
                image_path = os.path.join(images_folder, image_name_ori + ext)
                if os.path.exists(image_path):
                    break  # 找到匹配的文件后跳出循环
            else:
                raise FileNotFoundError(f"No image found for {image_name_ori} with .png or .jpg extension in {images_folder}")
            
            # image_path = os.path.join(images_folder, image_name_ori+'.png')
            image = Image.open(image_path)
            resized_image_rgb = PILtoTensor(image)
            gt_image = resized_image_rgb[:3, ...]
            
            # alpha
            alpha_path = os.path.join(alpha_folder, image_name_ori+'.jpg')
            alpha = Image.open(alpha_path)
            alpha = PILtoTensor(alpha)

            # # if add head mask
            head_mask_path = os.path.join(parsing_folder, image_name_ori+'_neckhead.png')
            head_mask = Image.open(head_mask_path)
            head_mask = PILtoTensor(head_mask)
            gt_image = gt_image * alpha + self.bg_image * (1 - alpha)
            gt_image = gt_image * head_mask + self.bg_image * (1 - head_mask)

            # mouth mask
            mouth_mask_path = os.path.join(parsing_folder, image_name_ori+'_mouth.png') # jpg
            mouth_mask = Image.open(mouth_mask_path)
            mouth_mask = PILtoTensor(mouth_mask)
            
            # # old camera
            # camera_indiv = Camera(colmap_id=frame_id, R=R, T=T, 
            #                     FoVx=FovX, FoVy=FovY, 
            #                     image=gt_image, head_mask=head_mask, mouth_mask=mouth_mask,
            #                     exp_param=exp_param, eyes_pose=eyes_pose, eyelids=eyelids, jaw_pose=jaw_pose,
            #                     image_name=image_name_mica, uid=frame_id, data_device=device)
            
            # # emoca camera
            # camera_indiv = Camera(colmap_id=frame_id, R=R, T=T, 
            #                     FoVx=FovX, FoVy=FovY, 
            #                     image=gt_image, head_mask=head_mask, mouth_mask=mouth_mask,
            #                     exp_param=exp_param, pose_param=pose_param, detail=detail,
            #                     image_name=image_name_mica, uid=frame_id, data_device=device)
            
            # smirk camera
            camera_indiv = Camera_smirk(colmap_id=frame_id, R=R, T=T, 
                                FoVx=FovX, FoVy=FovY, 
                                image=gt_image, head_mask=head_mask, mouth_mask=mouth_mask,
                                exp_param=exp_param, pose_param=pose_param, eyelids=eyelids, jaw_pose=jaw_pose,
                                image_name=image_name_mica, uid=frame_id, data_device=device)
            self.cameras.append(camera_indiv)
    
    def getCameras(self):
        return self.cameras



    
class Scene_mica_smirk_light:
    def __init__(self, datadir, mica_datadir, train_type, white_background, device, id_name = None, quan_bits = 0):
        ## train_type: 0 for train, 1 for test, 2 for eval
        frame_delta = 1 
        self.images_folder = os.path.join(datadir, "imgs")
        self.parsing_folder = os.path.join(datadir, "parsing")
        self.alpha_folder = os.path.join(datadir, "alpha")
        
        self.bg_image = torch.zeros((3, 512, 512))
        if white_background:
            self.bg_image[:, :, :] = 1
        else:
            self.bg_image[1, :, :] = 1

        mica_ckpt_dir = os.path.join(mica_datadir, 'checkpoint_offset')
        self.N_frames = len(os.listdir(mica_ckpt_dir))
        self.cameras = []
        
        # 设置数据范围
        test_num = 500
        eval_num = 50
        max_train_num = 2000
        train_num = min(max_train_num, self.N_frames - test_num)
        
        # 读取初始帧获取基本参数
        ckpt_path = os.path.join(mica_ckpt_dir, '00000.frame')
        payload = torch.load(ckpt_path)
        flame_params = payload['flame']
        self.shape_param = torch.as_tensor(flame_params['shape'])
        
        # 设置相机参数
        orig_w, orig_h = payload['img_size']
        K = payload['opencv']['K'][0]
        fl_x = K[0, 0]
        fl_y = K[1, 1]
        FovY = focal2fov(fl_y, orig_h)
        FovX = focal2fov(fl_x, orig_w)
        
        # 设置数据范围
        if train_type == 0:
            range_down = 0
            range_up = train_num + range_down
        elif train_type == 1:
            range_down = self.N_frames - test_num
            range_up = self.N_frames
        elif train_type == 2:
            range_down = self.N_frames - eval_num
            range_up = self.N_frames
        else:  # train_type == 3
            range_down = 0
            range_up = 1000
            
        # 如果使用量化，加载量化参数
        self.quan_bits = quan_bits
        if quan_bits != 0:
            print("using quantization", quan_bits,"bits, current id_name: ", id_name)
            max_flame_path = os.path.join(mica_datadir, f'{id_name}_max.frame')
            min_flame_path = os.path.join(mica_datadir, f'{id_name}_min.frame') 
            self.max_flame = torch.load(max_flame_path)
            self.min_flame = torch.load(min_flame_path)
            
            self.max_opencv_t = torch.tensor([[0.2,0.3,5]], dtype=torch.float32)
            self.min_opencv_t = torch.tensor([[-0.2,-0.3,0.5]], dtype=torch.float32)

        # 遍历所有帧
        for frame_id in tqdm(range(range_down, range_up)):
            image_name_mica = str(frame_id).zfill(5)
            image_name_ori = str(frame_id+frame_delta).zfill(5)
            ckpt_path = os.path.join(mica_ckpt_dir, image_name_mica+'.frame')
            
            # 读取FLAME参数和相机参数
            payload = torch.load(ckpt_path)
            flame_params = payload['flame']
            opencv = payload['opencv']
            
            # 处理FLAME参数
            params = {
                'exp_param': torch.as_tensor(flame_params['exp']),
                'pose_param': torch.as_tensor(flame_params['pose']),
                'eyelids': torch.as_tensor(flame_params['eyelids']),
                'jaw_pose': torch.as_tensor(flame_params['jaw'])
            }
            
            # 如果需要量化，进行参数量化
            if quan_bits != 0:
                params = self._quantize_params(params, opencv)
                w2cR = opencv['R'][0]
                w2cT = opencv['t'][0]
            else:
                w2cR = opencv['R'][0]
                w2cT = opencv['t'][0]
            
            R = np.transpose(w2cR)
            T = w2cT
            
            # 创建轻量级相机对象，只存储路径和参数
            camera_indiv = Camera_smirk_light(
                colmap_id=frame_id,
                R=R, T=T,
                FoVx=FovX, FoVy=FovY,
                image_path=self._get_image_path(image_name_ori),
                head_mask_path=self._get_head_mask_path(image_name_ori),
                mouth_mask_path=self._get_mouth_mask_path(image_name_ori),
                alpha_path=self._get_alpha_path(image_name_ori),
                **params,
                image_name=image_name_mica,
                uid=frame_id,
                data_device=device,
                bg_image=self.bg_image
            )
            self.cameras.append(camera_indiv)
    
    def _get_image_path(self, image_name):
        for ext in ['.png', '.jpg']:
            path = os.path.join(self.images_folder, image_name + ext)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No image found for {image_name}")
    
    def _get_head_mask_path(self, image_name):
        return os.path.join(self.parsing_folder, f"{image_name}_neckhead.png")
    
    def _get_mouth_mask_path(self, image_name):
        return os.path.join(self.parsing_folder, f"{image_name}_mouth.png")
    
    def _get_alpha_path(self, image_name):
        return os.path.join(self.alpha_folder, f"{image_name}.jpg")
    
    def _quantize_params(self, params, opencv):
        """量化所有参数"""
        if self.quan_bits == 0:
            return params
            
        # 量化表情参数
        params['exp_param'] = self._quantize_and_dequantize(
            params['exp_param'], 
            self.max_flame['flame']['exp'],
            self.min_flame['flame']['exp']
        )
        
        # 量化pose参数
        params['pose_param'] = self._quantize_and_dequantize(
            params['pose_param'],
            self.max_flame['flame']['pose'],
            self.min_flame['flame']['pose']
        )
        
        # 量化jaw参数
        params['jaw_pose'] = self._quantize_and_dequantize(
            params['jaw_pose'],
            self.max_flame['flame']['jaw'],
            self.min_flame['flame']['jaw']
        )
        
        # 量化eyelids参数
        params['eyelids'] = self._quantize_and_dequantize(
            params['eyelids'],
            torch.tensor([1,1]),
            torch.tensor([0,0])
        )
        
        # 量化opencv参数
        opencv_t = self._quantize_and_dequantize(
            torch.as_tensor(opencv['t'][0]),
            self.max_opencv_t,
            self.min_opencv_t
        )
        opencv['t'][0] = opencv_t
        
        # 量化R矩阵
        R_matrix = torch.as_tensor(opencv['R'][0])
        R_axis_angle = matrix_to_axis_angle(R_matrix)
        R_axis_angle = self._quantize_and_dequantize(
            R_axis_angle,
            torch.tensor(np.pi),
            torch.tensor(-np.pi)
        )
        opencv['R'][0] = axis_angle_to_matrix(R_axis_angle)
        
        return params
    
    def _quantize_and_dequantize(self, param, max_val, min_val):
        """量化并反量化参数"""
        quantized = quantize(param, max_val, min_val, self.quan_bits)
        return dequantize(quantized, max_val, min_val, self.quan_bits)
    
    def getCameras(self):
        return self.cameras
