import os, sys
import random
import json
from PIL import Image
import torch
import math
import numpy as np
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera,Camera_smirk,Camera_emoca
from arguments import ModelParams
from utils.general_utils import PILtoTensor
from utils.graphics_utils import focal2fov

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, axis_angle_to_matrix,euler_angles_to_matrix,matrix_to_axis_angle

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
            range_down = 300  # 从500帧开始，有的人前面的数据不好
            range_up = train_num + range_down
        if train_type == 1:
            range_down = self.N_frames - test_num
            range_up = self.N_frames
            # range_down = 10
            # range_up = 11
        if train_type == 2:
            range_down = self.N_frames - eval_num
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
            

            oepncv = payload['opencv']
            oepncv['t'] = np.array([[ 7.9078128e-04, -1.0053458e-02,  1.5682145e+00]], dtype=np.float32)
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
        if not white_background:
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
            # range_down = 10
            # range_up = 11
        if train_type == 2:
            range_down = self.N_frames - eval_num
            range_up = self.N_frames
            # range_up = range_down +1
            
        if train_type == 3:
            range_down = 291
            # range_up = self.N_frames
            range_up = 292
            
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
            oepncv = payload['opencv']
            
            exp_param = torch.as_tensor(flame_params['exp'])
            pose_param = torch.as_tensor(flame_params['pose'])
            detail = torch.as_tensor(flame_params['detail'])
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
                
                
                # shape_param = torch.as_tensor(flame_params['shape'])
                # camera_params = torch.as_tensor(np.array([[1.0019892e+01, 1.4389530e-03, 3.7328370e-02]]))

                # oepncv = payload['opencv']
                # oepncv['t'] = np.array([[ 7.9078128e-04, -1.0053458e-02,  1.5682145e+00]], dtype=np.float32)
                # oepncv['t'] = np.array([[ 7.9078128e-04, 20e-02,  1.5682145e+00]], dtype=np.float32)
            
                
            
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

        mica_ckpt_dir = os.path.join(mica_datadir, 'checkpoint_offset')
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
            range_down = 300
            range_up = train_num + range_down
        if train_type == 1:
            range_down = self.N_frames - test_num
            range_up = self.N_frames
            # range_down = 10
            # range_up = 11
        if train_type == 2:
            range_down = self.N_frames - eval_num
            range_up = self.N_frames
            

        for frame_id in tqdm(range(range_down, range_up)):
            image_name_mica = str(frame_id).zfill(5) # obey mica tracking
            image_name_ori = str(frame_id+frame_delta).zfill(5)
            ckpt_path = os.path.join(mica_ckpt_dir, image_name_mica+'.frame')
            payload = torch.load(ckpt_path)
            
            flame_params = payload['flame']
            
            # # ########### old training code
            # exp_param = torch.as_tensor(flame_params['exp'])
            # eyes_pose = torch.as_tensor(flame_params['eyes'])
            # eyelids = torch.as_tensor(flame_params['eyelids'])
            # jaw_pose = torch.as_tensor(flame_params['jaw'])
            
            # ########## emoca training code
            # exp_param = torch.as_tensor(flame_params['exp'])
            # pose_param = torch.as_tensor(flame_params['pose'])
            # detail = torch.as_tensor(flame_params['detail'])
            
            ############ smirk training code
            exp_param = torch.as_tensor(flame_params['exp'])
            pose_param = torch.as_tensor(flame_params['pose'])
            # eyes_pose = torch.as_tensor(flame_params['eyes'])
            eyelids = torch.as_tensor(flame_params['eyelids'])
            jaw_pose = torch.as_tensor(flame_params['jaw'])

            oepncv = payload['opencv']
            # cam  = payload['camera']['cam']
            
            w2cR = oepncv['R'][0] # mica opencv
            w2cT = oepncv['t'][0]
            # w2cR = np.array([[1,0,0],[0,-1,0],[0,0,-1]]) # smirk cam
            
            # w2cT = cam[0]
            # w2cT[0] = cam[0][1]
            # w2cT[1] = -cam[0][2]
            # # w2cT[2] = -0.0661* cam[0][0] + 1.973
            # w2cT[2] = (1/cam[0][0])*9.629
            
            
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



    
