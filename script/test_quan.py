import torch
import pprint
import numpy as np

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, axis_angle_to_matrix,euler_angles_to_matrix,matrix_to_axis_angle

def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)

# 量化
# region  

# path1 = "/home/xylem/IBC24/FlashAvatar-code/metrical-tracker/output/Obama/checkpoint/02399.frame"
# path2 = "/home/xylem/IBC24/smirk/output/Obama2/02399.frame"
# path3 = "/home/xylem/IBC24/emoca/gdl_apps/EMOCA/image_output/EMOCA_v2_lr_mse_20/id4/00021.frame"
# path4 = "/home/xylem/IBC24/emoca/gdl_apps/EMOCA/image_output/EMOCA_v2_lr_mse_20/Obama_wo_crop/00020.frame"
# frame1 = torch.load(path1)
# frame2 = torch.load(path3)

# flame_params = frame1['flame']
# # 获取数据范围（可以根据数据动态计算或设定固定范围）
# data = flame_params['exp']
# data_min = np.min(data)  # 数据的最小值
# data_max = np.max(data)  # 数据的最大值
# print("data_min",data_min)
# print("data_max",data_max)
# # 定义量化与反量化函数
# def quantize(data, bit):
#     """量化函数"""
#     levels = 2**bit - 1  # 离散等级数
#     q_data = np.round((data - data_min) / (data_max - data_min) * levels)  # 量化
#     return q_data, levels  # 返回量化数据和离散等级数

# def dequantize(q_data, levels, bit):
#     """反量化函数"""
#     d_data = q_data / levels * (data_max - data_min) + data_min  # 反量化
#     return d_data

# # 8-bit 量化与反量化
# q_data_8bit, levels_8bit = quantize(data, 8)
# print("levels_8bit",levels_8bit)
# d_data_8bit = dequantize(q_data_8bit, levels_8bit, 8)

# # 10-bit 量化与反量化
# q_data_10bit, levels_10bit = quantize(data, 10)
# print("levels_10bit",levels_10bit)
# d_data_10bit = dequantize(q_data_10bit, levels_10bit, 10)

# # 打印结果
# print("Original Data:\n", data)
# # print("8-bit Quantized Data:\n", q_data_8bit)
# print("8-bit Dequantized Data:\n", d_data_8bit)
# # print("10-bit Quantized Data:\n", q_data_10bit)
# print("10-bit Dequantized Data:\n", d_data_10bit)

# endregion


# emoca 参数量化，包括表情参数，姿势参数，以及旋转矩阵R， 可用。
# region
# import os
# import torch
# import numpy as np


# id_list = ["id4_emoca", "id5_emoca", "id7_emoca", "id8_emoca", "Obama_emoca", "ljl_wo_glass_emoca"]

# for id_name in id_list:
#     # 目标路径
#     path = f"/home/xylem/IBC24/FlashAvatar-code/metrical-tracker/output/{id_name}/checkpoint_offset"
#     root_path = f"/home/xylem/IBC24/FlashAvatar-code/metrical-tracker/output/{id_name}"
#     # 检查路径是否存在
#     if not os.path.exists(path):
#         print(f"Path not found: {path}")
#         continue
    
#     # 初始化全局最大值和最小值
#     global_max = None  # 用于存储每个维度的最大值
#     global_min = None  # 用于存储每个维度的最小值
    
#     pose_max = None
#     pose_min = None
    
#     R_min = None
#     R_max = None

#     # 遍历目录下所有 .frame 文件
#     for filename in os.listdir(path):
#         if filename.endswith('.frame'):  # 检查是否为 .frame 文件
#             file_path = os.path.join(path, filename)
            
#             try:
#                 # 加载文件
#                 frame = torch.load(file_path)
#                 # 提取 flame_params['exp']
#                 data = frame['flame']['exp']  # 假设 data 是一个二维 numpy 数组或 PyTorch 张量
#                 # 转为 numpy 以便后续处理
#                 data = data.numpy() if isinstance(data, torch.Tensor) else np.array(data)

#                 # 初始化 global_min 和 global_max
#                 if global_max is None:
#                     global_max = np.copy(data)
#                     global_min = np.copy(data)
#                 else:
#                     # 更新每个维度的最大值和最小值
#                     global_max = np.maximum(global_max, data)
#                     global_min = np.minimum(global_min, data)
                    
                
#                 # 处理pose
#                 data_pose = frame['flame']['pose']
#                 data_pose = data_pose.numpy() if isinstance(data_pose, torch.Tensor) else np.array(data_pose)
#                 if pose_max is None:
#                     pose_max = np.copy(data_pose)
#                     pose_min = np.copy(data_pose)
#                 else:
#                     pose_max = np.maximum(data_pose, pose_max)
#                     pose_min = np.minimum(data_pose, pose_min)
                    
#                 # 处理 R
#                 R_matrix = to_tensor(frame['opencv']['R'])
#                 R_angle = matrix_to_axis_angle(R_matrix)
#                 R_angle = R_angle.numpy() if isinstance(R_angle, torch.Tensor) else np.array(R_angle)
#                 if R_max is None:
#                     R_max = np.copy(R_angle)
#                     R_min = np.copy(R_angle)
#                 else:
#                     R_max = np.maximum(R_max, R_angle)
#                     R_min = np.minimum(R_min, R_angle)
                
#             except Exception as e:
#                 print(f"Error processing file {file_path}: {e}")

#     # 保存最大值和最小值到 .frame 文件
#     if global_max is not None and global_min is not None:
#         max_frame = {'flame': {'exp': global_max, 'pose': pose_max}, 'opencv': {'R': R_max}}
#         min_frame = {'flame': {'exp': global_min, 'pose': pose_min}, 'opencv': {'R': R_min}}
        
#         # 保存为 frame 文件
#         max_output_path = os.path.join(root_path, f"{id_name}_max.frame")
#         min_output_path = os.path.join(root_path, f"{id_name}_min.frame")
#         torch.save(max_frame, max_output_path)
#         torch.save(min_frame, min_output_path)
#         print(f"Saved max to: {max_output_path}")
#         print(f"Saved min to: {min_output_path}")
#     else:
#         print(f"No valid .frame files found in {path}")

#endregion


# smirk 参数量化，获取最大值最小值。包括表情参数，姿势参数，以及旋转矩阵R， 可用。
# region
import os
import torch
import numpy as np


id_list = ["id4_emoca", "id5_emoca", "id7_emoca", "id8_emoca", "Obama_emoca", "ljl_wo_glass_emoca"]
smirk_id_list = ["id4_smirk", "id5_smirk", "id7_smirk", "id8_smirk", "Obama_smirk", "ljl_wo_glass_smirk"]
smirk_id_list = ["ljl_smirk","xj_new_smirk"]
for id_name in smirk_id_list:
    # 目标路径
    path = f"/home/xylem/IBC24/FlashAvatar-code/metrical-tracker/output/{id_name}/checkpoint_offset"
    root_path = f"/home/xylem/IBC24/FlashAvatar-code/metrical-tracker/output/{id_name}"
    # 检查路径是否存在
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        continue
    
    # 初始化全局最大值和最小值
    global_max = None  # 用于存储每个维度的最大值
    global_min = None  # 用于存储每个维度的最小值
    
    pose_max = None
    pose_min = None
    
    R_min = None
    R_max = None
    
    jaw_max =None
    jaw_min = None

    # 遍历目录下所有 .frame 文件
    for filename in os.listdir(path):
        if filename.endswith('.frame'):  # 检查是否为 .frame 文件
            file_path = os.path.join(path, filename)
            
            try:
                # 加载文件
                frame = torch.load(file_path)
                # 提取 flame_params['exp']
                data = frame['flame']['exp']  # 假设 data 是一个二维 numpy 数组或 PyTorch 张量
                # 转为 numpy 以便后续处理
                data = data.numpy() if isinstance(data, torch.Tensor) else np.array(data)

                # 初始化 global_min 和 global_max
                if global_max is None:
                    global_max = np.copy(data)
                    global_min = np.copy(data)
                else:
                    # 更新每个维度的最大值和最小值
                    global_max = np.maximum(global_max, data)
                    global_min = np.minimum(global_min, data)
                    
                
                # 处理pose
                data_pose = frame['flame']['pose']
                data_pose = data_pose.numpy() if isinstance(data_pose, torch.Tensor) else np.array(data_pose)
                if pose_max is None:
                    pose_max = np.copy(data_pose)
                    pose_min = np.copy(data_pose)
                else:
                    pose_max = np.maximum(data_pose, pose_max)
                    pose_min = np.minimum(data_pose, pose_min)
                    
                # 处理jaw
                data_jaw = frame['flame']['jaw']
                data_jaw = data_jaw.numpy() if isinstance(data_jaw, torch.Tensor) else np.array(data_jaw)
                if jaw_max is None:
                    jaw_max = np.copy(data_jaw)
                    jaw_min = np.copy(data_jaw)
                else:
                    jaw_max = np.maximum(data_jaw, jaw_max)
                    jaw_min = np.minimum(data_jaw, jaw_min)
                    
                # 处理 R
                R_matrix = to_tensor(frame['opencv']['R'])
                R_angle = matrix_to_axis_angle(R_matrix)
                R_angle = R_angle.numpy() if isinstance(R_angle, torch.Tensor) else np.array(R_angle)
                if R_max is None:
                    R_max = np.copy(R_angle)
                    R_min = np.copy(R_angle)
                else:
                    R_max = np.maximum(R_max, R_angle)
                    R_min = np.minimum(R_min, R_angle)
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    # 保存最大值和最小值到 .frame 文件
    if global_max is not None and global_min is not None:
        max_frame = {'flame': {'exp': global_max, 'pose': pose_max, 'jaw':jaw_max}, 'opencv': {'R': R_max}}
        min_frame = {'flame': {'exp': global_min, 'pose': pose_min, 'jaw':jaw_min}, 'opencv': {'R': R_min}}
        
        # 保存为 frame 文件
        max_output_path = os.path.join(root_path, f"{id_name}_max.frame")
        min_output_path = os.path.join(root_path, f"{id_name}_min.frame")
        torch.save(max_frame, max_output_path)
        torch.save(min_frame, min_output_path)
        print(f"Saved max to: {max_output_path}")
        print(f"Saved min to: {min_output_path}")
    else:
        print(f"No valid .frame files found in {path}")

#endregion


# # mica 参数量化，获取最大值最小值。包括表情参数，姿势参数，以及旋转矩阵R， 可用。
# region
# import os
# import torch
# import numpy as np


# # id_list = ["id4_emoca", "id5_emoca", "id7_emoca", "id8_emoca", "Obama_emoca", "ljl_wo_glass_emoca"]
# mica_id_list = ["id4", "id5_smirk", "id7_smirk", "id8_smirk", "Obama_smirk", "ljl_wo_glass_smirk"]
# mica_id_list = ["id2"]
# for id_name in mica_id_list:
#     # 目标路径
#     path = f"/home/xylem/IBC24/FlashAvatar-code/metrical-tracker/output/{id_name}/checkpoint"
#     root_path = f"/home/xylem/IBC24/FlashAvatar-code/metrical-tracker/output/{id_name}"
#     # 检查路径是否存在
#     if not os.path.exists(path):
#         print(f"Path not found: {path}")
#         continue
    
#     # 初始化全局最大值和最小值
#     param_max = None  # 用于存储每个维度的最大值
#     param_min = None  # 用于存储每个维度的最小值
    
#     pose_max = None
#     pose_min = None
    
#     R_min = None
#     R_max = None
    
#     jaw_max =None
#     jaw_min = None

#     # 遍历目录下所有 .frame 文件
#     for filename in os.listdir(path):
#         if filename.endswith('.frame'):  # 检查是否为 .frame 文件
#             file_path = os.path.join(path, filename)
            
#             try:
#                 # 加载文件
#                 frame = torch.load(file_path)
#                 # 提取 flame_params['exp']
#                 data = frame['flame']['exp']  # 假设 data 是一个二维 numpy 数组或 PyTorch 张量
#                 # 转为 numpy 以便后续处理
#                 data = data.numpy() if isinstance(data, torch.Tensor) else np.array(data)

#                 # 初始化 global_min 和 global_max
#                 if param_max is None:
#                     param_max = np.copy(data)
#                     param_min = np.copy(data)
#                 else:
#                     # 更新每个维度的最大值和最小值
#                     param_max = np.maximum(param_max, data)
#                     param_min = np.minimum(param_min, data)
                    
                
#                 # 处理pose
#                 data_pose = frame['flame']['eyes']
#                 data_pose = data_pose.numpy() if isinstance(data_pose, torch.Tensor) else np.array(data_pose)
#                 if pose_max is None:
#                     pose_max = np.copy(data_pose)
#                     pose_min = np.copy(data_pose)
#                 else:
#                     pose_max = np.maximum(data_pose, pose_max)
#                     pose_min = np.minimum(data_pose, pose_min)
                    
#                 # 处理jaw
#                 data_jaw = frame['flame']['jaw']
#                 data_jaw = data_jaw.numpy() if isinstance(data_jaw, torch.Tensor) else np.array(data_jaw)
#                 if jaw_max is None:
#                     jaw_max = np.copy(data_jaw)
#                     jaw_min = np.copy(data_jaw)
#                 else:
#                     jaw_max = np.maximum(data_jaw, jaw_max)
#                     jaw_min = np.minimum(data_jaw, jaw_min)
                    
#                 # 处理 R
#                 R_matrix = to_tensor(frame['opencv']['R'])
#                 R_angle = matrix_to_axis_angle(R_matrix)
#                 R_angle = R_angle.numpy() if isinstance(R_angle, torch.Tensor) else np.array(R_angle)
#                 if R_max is None:
#                     R_max = np.copy(R_angle)
#                     R_min = np.copy(R_angle)
#                 else:
#                     R_max = np.maximum(R_max, R_angle)
#                     R_min = np.minimum(R_min, R_angle)
                
#             except Exception as e:
#                 print(f"Error processing file {file_path}: {e}")

#     # 保存最大值和最小值到 .frame 文件
#     if param_max is not None and param_min is not None:
#         max_frame = {'flame': {'exp': param_max, 'eyes': pose_max, 'jaw':jaw_max}, 'opencv': {'R': R_max}}
#         min_frame = {'flame': {'exp': param_min, 'eyes': pose_min, 'jaw':jaw_min}, 'opencv': {'R': R_min}}
        
#         # 保存为 frame 文件
#         max_output_path = os.path.join(root_path, f"{id_name}_max.frame")
#         min_output_path = os.path.join(root_path, f"{id_name}_min.frame")
#         torch.save(max_frame, max_output_path)
#         torch.save(min_frame, min_output_path)
#         print(f"Saved max to: {max_output_path}")
#         print(f"Saved min to: {min_output_path}")
#     else:
#         print(f"No valid .frame files found in {path}")

# endregion