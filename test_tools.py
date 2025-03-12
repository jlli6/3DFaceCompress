import pickle
import os
def divide_pkl():
    path = "/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/ljl_smirk/log_smirk_eagles_150000_lpips_no_xyz_entropy_layer4/ckpt/point_cloud_compressed60000.pkl"
    with open(path,'rb') as f:
        data = pickle.load(f)
        latents = data['latents']
        decoder_state_dict = data['decoder_state_dict']
        decoder_args = data['decoder_args']
    
    # 分别存储
    output_path = "/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/ljl_smirk/log_smirk_eagles_150000_lpips_no_xyz_entropy_layer4/output"
    os.makedirs(output_path,exist_ok=True)
    with open(os.path.join(output_path,"latents.pkl"),'wb') as f:
            pickle.dump({
                         'latents': latents,
            }, f)
    with open(os.path.join(output_path,"decoder_state_dict.pkl"),'wb') as f:
            pickle.dump({
                         'decoder_state_dict': decoder_state_dict,
            }, f)
    with open(os.path.join(output_path,"decoder_args.pkl"),'wb') as f:
            pickle.dump({
                         'decoder_args': decoder_args,
            }, f)

        # for attribute in latents:
        #     if isinstance(latents[attribute], CompressedLatents):
        #         assert isinstance(self.latent_decoders[attribute], LatentDecoder)
        #         self.latent_decoders[attribute] = LatentDecoder(**decoder_args[attribute]).cuda()
        #         self.latent_decoders[attribute].load_state_dict(decoder_state_dict[attribute])
        #         self._latents[attribute] = nn.Parameter(latents[attribute].uncompress().cuda().requires_grad_(True))
        #     else:
        #         self._latents[attribute] = nn.Parameter(latents[attribute].cuda().requires_grad_(True))

        # self.active_sh_degree = self.max_sh_degree
  
def video_psnr():
        
        import cv2
        import numpy as np      
        # 比较两个视频的psnr
        video1_path = "/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/Obama_smirk/log_smirk_eagles_150000/rec.avi"
        video2_path = "/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/Obama_smirk/log_smirk_eagles_150000/ori.avi"

        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)

        if not cap1.isOpened() or not cap2.isOpened():
                print("无法打开视频文件")
        else:
                psnr_values = []
                while True:
                        ret1, frame1 = cap1.read()
                        ret2, frame2 = cap2.read()

                        if not ret1 or not ret2:
                                break

                        psnr = cv2.PSNR(frame1, frame2)
                        psnr_values.append(psnr)

                cap1.release()
                cap2.release()

                average_psnr = np.mean(psnr_values)
                print(f"平均PSNR: {average_psnr}")     
                
       
# 测试mlp相关的压缩操作

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, hidden_layers=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fcs = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_layers-1)]
        )
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        # input: B,V,d
        batch_size, N_v, input_dim = input.shape
        input_ori = input.reshape(batch_size*N_v, -1)
        h = input_ori
        for i, l in enumerate(self.fcs):
            h = self.fcs[i](h)
            h = F.relu(h)
        output = self.output_linear(h)
        output = output.reshape(batch_size, N_v, -1)

        return output  
    
import matplotlib.pyplot as plt
def plot_weights_distribution(model, save_path):
    # 将模型的每一层的权重绘制为直方图
    plt.figure(figsize=(12, 8))
    
    # 遍历每一层的权重
    for idx, layer in enumerate(model.fcs):
        weights = layer.weight.data.numpy()  # 获取权重数据
        plt.subplot(2, 2, idx+1)
        plt.hist(weights.flatten(), bins=50, density=True, alpha=0.6, color='g')
        plt.title(f'Layer {idx+1} Weight Distribution')
        plt.xlabel('Weight Value')
        plt.ylabel('Density')

    # 最后一层输出层的权重
    weights = model.output_linear.weight.data.numpy()
    plt.subplot(2, 2, 4)
    plt.hist(weights.flatten(), bins=50, density=True, alpha=0.6, color='b')
    plt.title(f'Output Layer Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Density')

    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()
  

def prune_mlp(mlp, pruning_ratio=0.8):
    # 剪枝比例设置为0.8，意味着保留20%的参数
    for layer in mlp.fcs:
        prune.l1_unstructured(layer, name="weight", amount=pruning_ratio)
    prune.l1_unstructured(mlp.output_linear, name="weight", amount=pruning_ratio)

    # 清理剪枝后不需要的部分
    for layer in mlp.fcs:
        prune.remove(layer, 'weight')
    prune.remove(mlp.output_linear, 'weight')

    return mlp

def test_mlp():
    # 创建模型
    mlp = MLP(input_dim=51+55, output_dim=10, hidden_dim=256, hidden_layers=4)
    
    # 加载训练好的模型
    # mlp_path = "/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/Obama_smirk/log_smirk_eagles_150000_mlp4/ckpt/chkpnt50000_mlp.pth"
    mlp_path = "/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/Obama_smirk/log_smirk_eagles_150000_mlp4_rest141_purn_progressive/ckpt/chkpnt150000_mlp.pth"
    mlp_checkpoint = torch.load(mlp_path)
    if len(mlp_checkpoint) == 3:  # 新格式包含best_psnr
        (model_params, first_iter, best_psnr) = mlp_checkpoint
    elif len(mlp_checkpoint) == 2:  # 旧格式不包含best_psnr
        (model_params, first_iter) = mlp_checkpoint
        best_psnr = 0.0  # 使用默认值
    else:
        model_params = mlp_checkpoint
    (net_params, opt_params) = model_params
    mlp.load_state_dict(net_params)
    
    # 绘制每一层的权重分布
    plot_weights_distribution(mlp, "./mlp_test/mlp4_rest141_purn_one_5000.png")
    
    # 剪枝
    # mlp_pruned = prune_mlp(mlp, pruning_ratio=0.8)
    
    # # 保存量化后的模型
    # state_dict_fp16 = {k: v.half() for k, v in mlp.state_dict().items()}
    # torch.save(state_dict_fp16, "./mlp_test/mlp_pruned_fp16_layer4.pth")

# 执行测试
# test_mlp()

def zaro_ratio(model):
    # 查看剪枝后的模型信息
    total_params = 0
    total_zero_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        num_zero_params = (param == 0).sum().item()
        zero_ratio = num_zero_params / num_params

        total_params += num_params
        total_zero_params += num_zero_params

        print(f"{name}, Pruned: {num_zero_params} / {num_params} ({zero_ratio:.2%})")

    # 查看整个模型权重为 0 的比例
    overall_zero_ratio = total_zero_params / total_params
    print(f"\nOverall Pruned Ratio: {total_zero_params} / {total_params} ({overall_zero_ratio:.2%})")
    

def compress_mlp():
    # 从稠密格式加载模型
    dense_path = "/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/id2_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan10_exp_dim_50/ckpt/chkpnt145000_mlp.pth"
    mlp = MLP(input_dim=51+55, output_dim=10, hidden_dim=256, hidden_layers=6)
    mlp_checkpoint = torch.load(dense_path)
    if len(mlp_checkpoint) == 3:  # 新格式包含best_psnr
        (model_params, first_iter, best_psnr) = mlp_checkpoint
    elif len(mlp_checkpoint) == 2:  # 旧格式不包含best_psnr
        (model_params, first_iter) = mlp_checkpoint
        # best_psnr = 0.0  # 使用默认值
    else:
        model_params = mlp_checkpoint
    (net_params, opt_params) = model_params
    mlp.load_state_dict(net_params)
    
    
    zaro_ratio(mlp)
    # zero_params, total_params, zero_ratio = count_zero_weights(net_params)
    # print(f"零值参数数量: {zero_params}, 总参数数量: {total_params}, 零值比例: {zero_ratio}")
    
    # # 保存为密集格式
    # sparse_path = "./mlp_test/mlp_layer4_pruned_150000_sparse.pth"
    
    # # 假设 model 是剪枝后的模型
    # pruned_sparse_state_dict = convert_to_sparse(mlp)

    # # 保存为稀疏格式
    # torch.save(pruned_sparse_state_dict, sparse_path)

def save_pruned_model_with_mask_old(model, save_path):
    """ 记录掩码并保存剪枝后的 MLP,减少存储空间 """
    pruned_state_dict = {}
    
    # 分别创建三个字典存储不同类型的参数
    weight_masks = {}
    weight_values = {}
    biases = {}
    
    for name, param in model.state_dict().items():
        if "weight" in name:  # 只对权重进行掩码存储
            mask = param != 0  # 记录非零权重的位置
            weight_masks[name] = mask
            weight_values[name] = param[mask]  # 仅存储非零值
        else:
            biases[name] = param  # 存储偏置参数
    
    # 分别保存三种参数
    torch.save(weight_masks, f"{save_path}_masks.pt")
    torch.save(weight_values, f"{save_path}_values.pt") 
    torch.save(biases, f"{save_path}_biases.pt")
    
    # 同时保存完整的pruned_state_dict用于向后兼容
    pruned_state_dict.update({f"{k}_mask": v for k, v in weight_masks.items()})
    pruned_state_dict.update({f"{k}_values": v for k, v in weight_values.items()})
    pruned_state_dict.update(biases)
    torch.save(pruned_state_dict, save_path)
    
    # 分别保存权重mask,权重value,以及偏置
    
    # 
    print(f"剪枝后的模型已保存至: {save_path}")
    
import torch
import numpy as np

def save_pruned_model_with_mask(model, save_path):
    """ 使用 1-bit 存储掩码，并保存剪枝后的 MLP """
    pruned_state_dict = {}
    weight_masks = {}
    weight_values = {}
    biases = {}

    for name, param in model.state_dict().items():
        if "weight" in name:
            mask = (param != 0).cpu().numpy().astype(np.uint8)  # 转换为 uint8 以便打包
            packed_mask = np.packbits(mask)  # 1-bit 压缩
            weight_masks[name] = torch.tensor(packed_mask)  # 转回 Tensor 以便存储
            weight_values[name] = param[param != 0]  # 仅存非零权重
        else:
            biases[name] = param  # 存储偏置参数

    # 统一存储，减少 I/O
    torch.save({"masks": weight_masks, "values": weight_values, "biases": biases}, save_path+"_fp32.pth")


def save_pruned_model_with_mask_fp16(model, save_path):
    """ 记录掩码并保存剪枝后的 MLP，减少存储空间 """
    
    weight_masks = {}
    weight_values = {}
    biases = {}

    for name, param in model.state_dict().items():
        if "weight" in name:  # 只对权重进行掩码存储
            mask = (param != 0).cpu().numpy().astype(np.uint8)  # 转换为 uint8 以便打包
            packed_mask = np.packbits(mask)  # 1-bit 压缩
            weight_masks[name] = packed_mask
            weight_values[name] = param[param != 0].half()  # 仅存储非零权重，并转换为 fp16
        else:
            biases[name] = param.half()  # 直接转换为 fp16 并存储偏置参数
    
    # 分别保存 mask、权重值和偏置
    torch.save(weight_masks, f"{save_path}_masks_fp16.pt")      # 存储 1-bit mask
    torch.save(weight_values, f"{save_path}_values_fp16.pt")    # 存储 fp16 非零权重
    torch.save(biases, f"{save_path}_biases_fp16.pt")           # 存储 fp16 偏置
    
    # # 兼容性存储完整剪枝后的 state_dict
    # pruned_state_dict = {
    #     **{f"{k}_mask": v for k, v in weight_masks.items()},
    #     **{f"{k}_values": v for k, v in weight_values.items()},
    #     **biases
    # }
    
    torch.save({"masks": weight_masks, "values": weight_values, "biases": biases}, save_path+"_fp16.pth")
    # torch.save(pruned_state_dict, save_path+"_fp16.pth")
    
def small_mlp():

    # 加载你的模型
    dense_path = "/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/id2_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan8_exp_dim_10/ckpt/chkpnt150000_mlp.pth"
    mlp = MLP(input_dim=51+55, output_dim=10, hidden_dim=256, hidden_layers=6)
    mlp_checkpoint = torch.load(dense_path)

    # 解析模型参数
    if len(mlp_checkpoint) == 3:
        (model_params, first_iter, best_psnr) = mlp_checkpoint
    elif len(mlp_checkpoint) == 2:
        (model_params, first_iter) = mlp_checkpoint
    else:
        model_params = mlp_checkpoint

    (net_params, opt_params) = model_params
    mlp.load_state_dict(net_params)

    # 保存剪枝后的模型
    pruned_path = "./mlp_test/small_mlp/pruned_mlp.pth" 
    simple_path = "./mlp_test/small_mlp/simple_mlp.pth"  # 添加具体路径
    save_pruned_model_with_mask_fp16(mlp, pruned_path)
    
    torch.save(mlp.state_dict(),simple_path)

def load_pruned_model_with_mask(model, load_path):
    """ 载入剪枝后的 MLP 并恢复权重 """
    pruned_state_dict = torch.load(load_path)
    
    new_state_dict = {}
    for name, param in model.state_dict().items():
        if f"{name}_mask" in pruned_state_dict:  # 说明该权重已被剪枝
            mask = pruned_state_dict[f"{name}_mask"]
            restored_weight = torch.zeros_like(param)  # 先初始化为 0
            restored_weight[mask] = pruned_state_dict[f"{name}_values"]  # 只填充非零值
            new_state_dict[name] = restored_weight
        else:
            new_state_dict[name] = pruned_state_dict[name]  # 其他参数直接赋值

    model.load_state_dict(new_state_dict)
    print("剪枝后的模型已成功加载！")

    return model
def load_prund():
    # 重新加载剪枝后的模型
    loaded_mlp = MLP(input_dim=51+55, output_dim=10, hidden_dim=256, hidden_layers=6)
    pruned_path = "./mlp_test/pruned_mlp.pth"
    loaded_mlp = load_pruned_model_with_mask(loaded_mlp, pruned_path)
    zaro_ratio(loaded_mlp)


def count_zero_weights(state_dict):
    total_params = 0
    zero_params = 0

    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):  # 确保是张量
            total_params += param.numel()  # 总参数数量
            zero_params += (param == 0).sum().item()  # 统计零值数量

    zero_ratio = zero_params / total_params if total_params > 0 else 0
    return zero_params, total_params, zero_ratio


# 将模型权重转为稀疏格式
def convert_to_sparse(model):
    sparse_state_dict = {}
    for name, param in model.state_dict().items():
        if param.dim() > 1:  # 只处理多维权重（如卷积层、全连接层的权重）
            sparse_state_dict[name] = param.to_sparse()  # 转为稀疏张量
        else:
            sparse_state_dict[name] = param  # 对于偏置等一维参数，保持密集格式
    return sparse_state_dict

# 从稀疏格式加载模型
def load_sparse_model(model, sparse_path):
    sparse_state_dict = torch.load(sparse_path)
    dense_state_dict = {}
    
    for name, param in sparse_state_dict.items():
        if param.is_sparse:  # 如果是稀疏张量
            dense_state_dict[name] = param.to_dense()  # 转为密集张量
        else:
            dense_state_dict[name] = param
    
    # 加载到模型中
    model.load_state_dict(dense_state_dict)
    return model

def json_arrange():
    import json
    with open("/home/ljl/workspace/IBC24/FlashAvatar-code/flame/FLAME_masks/region_point_simple.json",'r') as f:
        data = json.load(f)
    complex_region = ['right_eyeball', 'no_home', 'left_eyeball', 'eye_region',  'lips']
    simple_region = ['scalp', 'face', 'forehead', 'neck', 'nose', 'right_ear', 'left_ear' ]
    complex_simlpe_dict = {"complex":[],"simple":[]}
    print(data.keys())
    for key  in data.keys():
        print("len of ",key,len(data[key]))
        if key in complex_region:
            complex_simlpe_dict["complex"].extend(data[key])
        elif key in simple_region:
            complex_simlpe_dict["simple"].extend(data[key])
    # print(complex_simlpe_dict)
    print("complex_region",len(complex_simlpe_dict["complex"]))
    print("simple_region",len(complex_simlpe_dict["simple"]))
    
    # 将字典写入json文件
    with open("/home/ljl/workspace/IBC24/FlashAvatar-code/flame/FLAME_masks/complex_simple_region.json",'w') as f:
        json.dump(complex_simlpe_dict,f)

if __name__ == "__main__":
#     divide_pkl()
    # test_mlp()
    # compress_mlp()
    small_mlp()
    # load_prund()
    # json_arrange()
 