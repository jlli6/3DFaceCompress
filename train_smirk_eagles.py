import os, sys 
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
import cv2
import lpips
import time
import json
import yaml
from collections import defaultdict

from scene import GaussianModel, Scene_mica,Scene_mica_smirk, GaussianModelSQ, Scene_mica_smirk_light
from src.deform_model import Deform_Model,Deform_Model_smirk
from gaussian_renderer import render, render_eagles
from arguments import ModelParams, PipelineParams, OptimizationParams, QuantizeParams
from utils.loss_utils import huber_loss, ssim
from utils.general_utils import normalize_for_percep, DecayScheduler

from compress.decoders import LatentDecoder
from compress.inf_loss import EntropyLoss
from collections import OrderedDict

from PIL import Image
from utils.general_utils import PILtoTensor
# import pytorch_ssim

# def calculate_psnr(img, gt):
#     mse = torch.mean((img - gt) ** 2)
#     return -10 * torch.log10(mse)

def calculate_psnr_on_test(scene_test, DeformModel, gaussians, bg_color, ppt, image_buffer):
    DeformModel.eval()
    
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)
    
    viewpoint = scene_test.getCameras().copy()
    codedict = {}
    codedict['shape'] = scene_test.shape_param.to(args.device)
    DeformModel.example_init(codedict)
    
    total_psnr = 0.0
    with torch.no_grad():
        for i, viewpoint_cam in enumerate(viewpoint):
            # 使用图像缓冲区加载图像
            image_buffer.load_to_camera(viewpoint_cam)
            
            codedict['expr'] = viewpoint_cam.exp_param
            codedict['pose'] = viewpoint_cam.pose_param
            codedict['eyelids'] = viewpoint_cam.eyelids
            codedict['jaw_pose'] = viewpoint_cam.jaw_pose
            
            verts_final, rot_delta, scale_coef = DeformModel.smirk_decode(codedict)
            gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])
            
            render_pkg = render_eagles(viewpoint_cam, gaussians, ppt, background)
            image = render_pkg["render"].clamp(0, 1)
        
            # 修改PSNR计算部分
            gt_image_np = (viewpoint_cam.original_image*255.).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
            image_np = (image*255.).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
            
            # 确保使用BGR格式
            gt_image_bgr = gt_image_np[:,:,::-1]  # RGB to BGR
            image_bgr = image_np[:,:,::-1]  # RGB to BGR
            
            # 使用OpenCV的PSNR计算
            psnr = cv2.PSNR(image_bgr, gt_image_bgr)
            total_psnr += psnr
            # # 计算PSNR
            # psnr = calculate_psnr(image, viewpoint_cam.original_image).mean().double()
            # print("psnr: ", psnr)
            
            # total_psnr += psnr
    
    avg_psnr = total_psnr / len(viewpoint)
    
    # 最后清理一次缓存
    torch.cuda.empty_cache()
    
    # 恢复训练模式
    DeformModel.train()
    
    return avg_psnr

def calculate_psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ImageBuffer:
    def __init__(self, device, white_background, image_size=512):
        # 预分配固定大小的显存
        self.image_buffer = torch.empty(3, image_size, image_size, device=device)  # RGB图像
        self.head_mask_buffer = torch.empty(3, image_size, image_size, device=device)  # mask
        self.mouth_mask_buffer = torch.empty(3, image_size, image_size, device=device)  # mask
        self.alpha_buffer = torch.empty(3, image_size, image_size, device=device)  # alpha
        self.bg_image = torch.empty(3, image_size, image_size, device=device)  # 临时图像缓存
        
        self.bg_image = torch.zeros((3, 512, 512), device=device)
        if  white_background:
            self.bg_image[:, :, :] = 1
        else:
            self.bg_image[1, :, :] = 1
        
    def load_to_camera(self, camera):
        
        
        """将图像数据加载到预分配的缓冲区，并设置相机引用"""
        # 加载原始图像
        image = PILtoTensor(Image.open(camera.image_path))
        self.image_buffer.copy_(image[:3, ...])  # 复制到临时缓存
        
        # 加载alpha
        alpha = PILtoTensor(Image.open(camera.alpha_path))
        self.alpha_buffer.copy_(alpha)
        
        # 加载head mask并应用
        head_mask = PILtoTensor(Image.open(camera.head_mask_path))
        self.head_mask_buffer.copy_(head_mask)
        
        # 加载mouth mask
        mouth_mask = PILtoTensor(Image.open(camera.mouth_mask_path))
        self.mouth_mask_buffer.copy_(mouth_mask)
        
        # 按照原代码的处理顺序应用mask
        self.image_buffer.copy_(self.image_buffer * self.alpha_buffer + self.bg_image * (1 - self.alpha_buffer))
        self.image_buffer.copy_(self.image_buffer * self.head_mask_buffer + self.bg_image * (1 - self.head_mask_buffer))
        
        # 设置相机引用到缓冲区
        camera.original_image = self.image_buffer
        camera.head_mask = self.head_mask_buffer
        camera.mouth_mask = self.mouth_mask_buffer


def add_loss_args(parser):
    parser.add_argument('--loss_type', type=str, default='lpips', choices=['lpips', 'ssim'],
                       help='Type of perceptual loss to use: lpips or ssim')
    return parser

class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
    def log(self, message):
        # 打印到终端
        print(message)
        # 写入日志文件
        with open(self.log_path, 'a') as f:
            f.write(message + '\n')

def training(args, lpt, opt, ppt, quantize):
    
    # 渐进式剪枝
    # mlp_purn_setting = {80000:0.4, 100000:0.6, 120000:0.7}
    mlp_purn_setting = {100000000:0.1}
    prune_iter = mlp_purn_setting.keys()
    
    batch_size = 1
    set_random_seed(args.seed)

    print("id_name:",args.idname, "exp_dim: ", args.exp_dim,"use_quan: ", args.quan_bit)
    ## deform model
    DeformModel = Deform_Model_smirk(args.device,
                                     exp_uesd_dim=args.exp_dim,mlp_layer=args.mlp_layer,hidden_dim=args.hidden_dim).to(args.device)
    
    DeformModel.training_setup()

    ## dataloader
    data_dir = os.path.join('/home/xylem/IBC24/FlashAvatar-code/dataset', args.idname)
    mica_datadir = os.path.join('/home/xylem/IBC24/FlashAvatar-code/metrical-tracker/output', args.idname)
    save_dir = os.path.join('dataset', args.idname)
    
    log_dir = os.path.join(save_dir, args.logname)
    train_dir = os.path.join(log_dir, 'train')
    model_dir = os.path.join(log_dir, 'ckpt')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建日志记录器
    logger = Logger(os.path.join(log_dir, 'training.log'))
    logger.log(f"Starting training with loss type: {args.loss_type}")
    
    # 初始化损失函数
    if args.loss_type == 'lpips':
        percep_module = lpips.LPIPS(net='vgg').to(args.device)
    # else:  # ssim
    #     ssim_module = pytorch_ssim.SSIM(window_size=11).to(args.device)
    
    # scene_test = Scene_mica_smirk_light(data_dir, mica_datadir, train_type=1, white_background=lpt.white_background, device = args.device,id_name=args.idname, quan_bits=args.quan_bit)
    # train process
    best_psnr = 0.0
    first_iter = 0
    
    quantize.use_shift = [bool(el) for el in quantize.use_shift]
    quantize.use_gumbel = [bool(el) for el in quantize.use_gumbel]
    quantize.gumbel_period = [bool(el) for el in quantize.gumbel_period]
    
    # gaussians = GaussianModel(lpt.sh_degree)
    # gaussians = GaussianModel(args.sh)
    gaussians = GaussianModelSQ(args.sh, quantize)
    distortion_loss = EntropyLoss(gaussians.prob_models, lambdas=gaussians.ent_lambdas, noise_freq=quantize.noise_freq)
    
    scene = Scene_mica_smirk(data_dir, mica_datadir, train_type=0, white_background=lpt.white_background, device = args.device,id_name=args.idname, quan_bits=args.quan_bit)
    
    temperature_scheds = OrderedDict()
    for i,param in enumerate(gaussians.param_names):
        temperature_scheds[param] = DecayScheduler(
                                        total_steps=opt.iterations+1,
                                        decay_name='exp',
                                        start=1.0,
                                        end=quantize.temperature[i],
                                        params={'temperature': quantize.temperature[i], 'decay_period': quantize.gumbel_period[i]},
                                        )
        
    gaussians.training_setup(opt)
    if args.start_checkpoint:
        print(f"Loading checkpoint from {args.start_checkpoint}")
        gaussian_path = os.path.join(model_dir, f"point_cloud_compressed{args.start_checkpoint}.pkl")
        mlp_path = os.path.join(model_dir, f"chkpnt{args.start_checkpoint}_mlp.pth")  
          
        mlp_checkpoint = torch.load(mlp_path)
        if len(mlp_checkpoint) == 3:  # 新格式包含best_psnr
            (model_params, first_iter, best_psnr) = mlp_checkpoint
        else:  # 旧格式不包含best_psnr
            (model_params, first_iter) = mlp_checkpoint
            best_psnr = 0.0  # 使用默认值
            
        DeformModel.restore(model_params)
        gaussians.load_compressed_pkl(gaussian_path)
        gaussians.decode_latents()
        
        
        print(f"Loaded checkpoint at iteration {first_iter} with best PSNR {best_psnr:.2f}")

    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)
    
    codedict = {}
    codedict['shape'] = scene.shape_param.to(args.device)
    DeformModel.example_init(codedict)

    viewpoint_stack = None
    first_iter += 1
    mid_num = 15_000 # 不用lpips损失
    entropy_loss_num = 30_000 # 开始使用entropy loss
    # entropy_loss_num = 0
    total_iterations = 150_000
    net_training_time = 0
    
    torch.cuda.synchronize()
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    torch.cuda.synchronize() 
    last_time = time.time()
    
    # 分配显存
    # image_buffer = ImageBuffer(args.device, lpt.white_background)
    
    # for iteration in range(first_iter, opt.iterations + 1):
    for iteration in range(first_iter, total_iterations + 1):
        
        if iteration in prune_iter:
            print(f"Pruning MLP at iteration {iteration} with ratio {mlp_purn_setting[iteration]}============")
            DeformModel.deformNet.prune_mlp(pruning_ratio=mlp_purn_setting[iteration])
            
        # torch.cuda.synchronize()
        # iter_start.record()
        
        gaussians.update_learning_rate(iteration, quantize)
        for i,param in enumerate(gaussians.param_names):
            gaussians.latent_decoders[param].temperature = temperature_scheds[param](iteration)
            gaussians.latent_decoders[param].use_gumbel = ((iteration / opt.iterations) < quantize.gumbel_period[i]) and quantize.use_gumbel[i]

        if (iteration-1) % 10 == 0:
            for i,param in enumerate(gaussians.param_names):
                if isinstance(gaussians.latent_decoders[param], LatentDecoder):
                    gaussians.latent_decoders[param].normalize(gaussians._latents[param])
        # Every 500 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getCameras().copy()
            random.shuffle(viewpoint_stack)
            # if len(viewpoint_stack)>3000:
            #     viewpoint_stack = viewpoint_stack[:3000]
            if len(viewpoint_stack)>2000:
                viewpoint_stack = viewpoint_stack[:2000]
        viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1)) 
        
        # image_buffer.load_to_camera(viewpoint_cam)
        frame_id = viewpoint_cam.uid
        
        # 在使用图像前加载图像数据
        # viewpoint_cam.load_images()
        
        #smirk deform
        codedict['expr'] = viewpoint_cam.exp_param
        codedict['pose'] = viewpoint_cam.pose_param
        codedict['eyelids'] = viewpoint_cam.eyelids
        codedict['jaw_pose'] = viewpoint_cam.jaw_pose 
        
        verts_final, rot_delta, scale_coef = DeformModel.smirk_decode(codedict)
        
        if iteration == 1:
            gaussians.create_from_verts(verts_final[0])
            gaussians.training_setup(opt)
        gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

        # Render
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=opt.use_amp):
            render_pkg = render_eagles(viewpoint_cam, gaussians, ppt, background)
            image = render_pkg["render"]
            
            # Loss
            gt_image = viewpoint_cam.original_image
            mouth_mask = viewpoint_cam.mouth_mask
            
            loss_huber = huber_loss(image, gt_image, 0.1) + 40*huber_loss(image*mouth_mask, gt_image*mouth_mask, 0.1)
            
            loss_G = 0.
            entropy_loss = 0.0   
            head_mask = viewpoint_cam.head_mask
            image_percep = normalize_for_percep(image*head_mask)
            gt_image_percep = normalize_for_percep(gt_image*head_mask)
            
            # entropy_loss, _ = distortion_loss.loss(gaussians._latents, iteration, is_val=(quantize.noise_freq == 0))
            if iteration>mid_num:
                if args.loss_type == 'lpips':
                    # LPIPS损失（值越小越好）
                    loss_G = torch.mean(percep_module.forward(image_percep, gt_image_percep)) * 0.05
                else:
                    # SSIM损失（值越大越好，所以用1-SSIM）
                    image_percep_batch = image_percep.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
                    gt_image_percep_batch = gt_image_percep.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
                    loss_G = (1 - ssim(image_percep_batch, gt_image_percep_batch)) * 0.05
            
             
            if iteration>entropy_loss_num:
                #entropy loss
                # loss = loss_huber*1 + loss_G*1+ entropy_loss*0.001
                loss = loss_huber*1 + loss_G*1+ entropy_loss*0.001
                # loss = entropy_loss
            else:
                loss = loss_huber*1 + loss_G*1
            
        

        loss.backward()

        
        
        with torch.no_grad():
            # 执行优化步骤
            if iteration < total_iterations:
                gaussians.optimizer.step()  # 更新高斯模型的优化器
                DeformModel.optimizer.step()  # 更新变形模型的优化器
                
                # 清零梯度
                gaussians.optimizer.zero_grad(set_to_none=True)  # 清零高斯模型的梯度
                DeformModel.optimizer.zero_grad(set_to_none=True)  # 清零变形模型的梯度
            
            # print loss
            if iteration % 500 == 0:
                torch.cuda.synchronize()
                current_time = time.time()
                time_cost = current_time - last_time
                last_time = current_time
                
                if iteration <= mid_num:
                    log_msg = f"step: {iteration}, huber: {loss_huber.item():.5f}, entropy_loss: {entropy_loss:.5f}, total loss: {loss.item():.5f}, time: {time_cost:.2f}s"
                else:
                    perceptual_term = "LPIPS" if args.loss_type == 'lpips' else "SSIM"
                    log_msg = f"step: {iteration}, huber: {loss_huber.item():.5f}, {perceptual_term}: {loss_G.item():.5f}, entropy_loss: {entropy_loss:.5f}, total loss: {loss.item():.5f}, time: {time_cost:.2f}s"
                logger.log(log_msg)
            
            # visualize results
            if iteration % 500 == 0 or iteration==1:
                save_image = np.zeros((args.image_res, args.image_res*2, 3))
                gt_image_np = (gt_image*255.).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                image = image.clamp(0, 1)
                image_np = (image*255.).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                save_image[:, :args.image_res, :] = gt_image_np
                save_image[:, args.image_res:, :] = image_np
                cv2.imwrite(os.path.join(train_dir, f"{iteration}.jpg"), save_image[:,:,[2,1,0]])
            
            # # 清理本次迭代的临时变量
            # del render_pkg, image, image_percep, gt_image_percep
                
            # save checkpoint
            # 保存checkpoint和评估PSNR
        if iteration % 5000 == 0:
            
            # PSNR计算不正确，先注释掉
            # logger.log(f"[ITER {iteration}] Evaluating PSNR")
            # current_psnr = calculate_psnr_on_test(scene_test, DeformModel, gaussians,bg_color,ppt,image_buffer)
            # logger.log(f"Current PSNR: {current_psnr:.2f}")
            
            current_psnr = 0.0
            # 保存压缩版本
            torch.save((DeformModel.capture(), iteration, current_psnr), 
                      os.path.join(model_dir, f"chkpnt{iteration}_mlp.pth"))
            gaussians.save_compressed_pkl_light(
                os.path.join(model_dir, f"point_cloud_compressed{iteration}.pkl"), 
                quantize)
            
            # 如果是最佳PSNR，保存best模型
            if current_psnr > best_psnr:
                best_psnr = current_psnr
                logger.log(f"New best PSNR: {best_psnr:.2f}")
                
                # 保存最佳模型
                torch.save((DeformModel.capture(), iteration, best_psnr), 
                          os.path.join(model_dir, "best_model_mlp.pth"))
                gaussians.save_compressed_pkl_light(
                    os.path.join(model_dir, "best_model_compressed.pkl"), 
                    quantize)    
                
    return net_training_time



def testing(args, lpt, opt, ppt, quantize):
    batch_size = 1
    set_random_seed(args.seed)

    ## deform model
    DeformModel = Deform_Model_smirk(args.device,
                                     exp_uesd_dim=args.exp_dim, mlp_layer=args.mlp_layer,hidden_dim=args.hidden_dim).to(args.device)
    
    DeformModel.training_setup()
    DeformModel.eval()

    ## dataloader
    data_dir = os.path.join('/home/xylem/IBC24/FlashAvatar-code/dataset', args.idname)
    mica_datadir = os.path.join('/home/xylem/IBC24/FlashAvatar-code/metrical-tracker/output', args.idname)
    save_dir = os.path.join('dataset', args.idname)
    logdir = os.path.join(save_dir, args.logname)
    
     # train process
    first_iter = 0
    
    quantize.use_shift = [bool(el) for el in quantize.use_shift]
    quantize.use_gumbel = [bool(el) for el in quantize.use_gumbel]
    quantize.gumbel_period = [bool(el) for el in quantize.gumbel_period]
    
    gaussians = GaussianModelSQ(args.sh, quantize)
    
    gaussians.training_setup(opt)
    
    
    if args.iteration:
        mlp_path = logdir + "/ckpt/chkpnt" + str(args.iteration) + "_mlp.pth"
        gaussian_path = logdir + "/ckpt/point_cloud_compressed" + str(args.iteration) + ".pkl"
        
        # 测试一下fp16的高斯参数和mlp参数
        # gaussian_path = "/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/Obama_smirk/log_smirk_eagles_150000/ckpt/separate/150000.pth"
        # mlp_path = "/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/Obama_smirk/log_smirk_eagles_150000/ckpt/fp16/mlp_150000"
        mlp_checkpoint = torch.load(mlp_path)
        if len(mlp_checkpoint) == 3:  # 新格式包含best_psnr
            (model_params, first_iter, best_psnr) = mlp_checkpoint
        elif len(mlp_checkpoint) == 2:  # 旧格式不包含best_psnr
            (model_params, first_iter) = mlp_checkpoint
            best_psnr = 0.0  # 使用默认值
        else:
            model_params = mlp_checkpoint
            
        # DeformModel.restore(model_params)
        DeformModel.restore(model_params)
        gaussians.load_compressed_pkl(gaussian_path)
        # gaussians.load_compressed_pkl_bit10(gaussian_path)
    
    gaussians.decode_latents()
    
    # 根据高斯区域来渲染不同区域的人脸
    region_mask_path = "/home/ljl/workspace/IBC24/FlashAvatar-code/flame/FLAME_masks/region_point_simple.json"
    region_mask = json.load(open(region_mask_path, "r"))
    # gaussian_idx_right_eyeball = region_mask["right_eyeball"]
    # print(gaussian_idx_right_eyeball)

    scene = Scene_mica_smirk(data_dir, mica_datadir, train_type=1, white_background=lpt.white_background,id_name=args.idname, quan_bits=args.quan_bit, device = args.device)
    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)
    
    
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # 对每隔区域的高斯都各自保存成不同的视频
    # out_list = {}
    # for region in region_mask.keys():
    #     vid_save_path = os.path.join(logdir, f'{region}.avi')
    #     out = cv2.VideoWriter(vid_save_path, fourcc, 30, (args.image_res*2, args.image_res), True)
    #     out_list[region] = out
        
    # 原本的保存
    vid_save_path = os.path.join(logdir, 'test.avi') 
    out = cv2.VideoWriter(vid_save_path, fourcc, 30, (args.image_res*2, args.image_res), True)
    
    # vid_save_path_rec = os.path.join(logdir, f'rec_{args.quan_bit}bit_exp_dim{args.exp_dim}_{args.chkpnt_number}_frame1000.avi')
    vid_save_path_rec = os.path.join(logdir, f'rec_{args.quan_bit}bit_exp_dim{args.exp_dim}_{args.chkpnt_number}_frame500.avi')
    # vid_save_path_rec = os.path.join(logdir, f'rec_{args.quan_bit}bit_exp_dim{args.exp_dim}_{args.chkpnt_number}_frame1000_best_compressed.avi')
    
    out_rec = cv2.VideoWriter(vid_save_path_rec, fourcc, 30, (args.image_res, args.image_res), True)
    
    # vid_save_path_ori = os.path.join(logdir, 'ori_frame1000.avi')
    vid_save_path_ori = os.path.join(logdir, 'ori_frame500.avi')
    out_ori = cv2.VideoWriter(vid_save_path_ori, fourcc, 30, (args.image_res, args.image_res), True)
    
    viewpoint = scene.getCameras().copy()
    codedict = {}
    codedict['shape'] = scene.shape_param.to(args.device)
    DeformModel.example_init(codedict)
    
    # gaussian_time = 0.
    # mlp_time = 0.
    
    total_num = 0
    total_time = 0
    for iteration in range(len(viewpoint)):
        
        
        # mlp_begin_time = time.time()
        viewpoint_cam = viewpoint[iteration]
        frame_id = viewpoint_cam.uid

        # # deform gaussians
        codedict['expr'] = viewpoint_cam.exp_param
        codedict['pose'] = viewpoint_cam.pose_param
        codedict['eyelids'] = viewpoint_cam.eyelids
        codedict['jaw_pose'] = viewpoint_cam.jaw_pose
        
        begin_time = time.time()
        # verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)
        verts_final, rot_delta, scale_coef = DeformModel.smirk_decode(codedict)
        gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])
        # mlp_time+=time.time()-mlp_begin_time
        # Render
        # gaussian_begin_time = time.time()
        
        # 对每个区域的高斯都保存成视频
        # for region in region_mask.keys():
        #     render_pkg = render_eagles(viewpoint_cam, gaussians, ppt, background, gaussian_region = region_mask[region])
        #     image= render_pkg["render"]
        #     image = image.clamp(0, 1)

        #     gt_image = viewpoint_cam.original_image
        #     save_image = np.zeros((args.image_res, args.image_res*2, 3))
        #     gt_image_np = (gt_image*255.).permute(1,2,0).detach().cpu().numpy()
        #     image_np = (image*255.).permute(1,2,0).detach().cpu().numpy()

        #     save_image[:, :args.image_res, :] = gt_image_np
        #     save_image[:, args.image_res:, :] = image_np
        #     save_image = save_image.astype(np.uint8)
        #     save_image = save_image[:,:,[2,1,0]]
            
        #     save_image_rec = image_np.astype(np.uint8)[:, :, [2,1,0]]
        #     save_image_ori = gt_image_np.astype(np.uint8)[:, :, [2,1,0]]
            
        #     out_list[region].write(save_image)
            
            
        render_pkg = render_eagles(viewpoint_cam, gaussians, ppt, background)
        if iteration>20:
            total_time+=time.time() - begin_time
            total_num += 1
        image= render_pkg["render"]
        image = image.clamp(0, 1)

        gt_image = viewpoint_cam.original_image
        save_image = np.zeros((args.image_res, args.image_res*2, 3))
        gt_image_np = (gt_image*255.).permute(1,2,0).detach().cpu().numpy()
        image_np = (image*255.).permute(1,2,0).detach().cpu().numpy()

        save_image[:, :args.image_res, :] = gt_image_np
        save_image[:, args.image_res:, :] = image_np
        save_image = save_image.astype(np.uint8)
        save_image = save_image[:,:,[2,1,0]]
        
        save_image_rec = image_np.astype(np.uint8)[:, :, [2,1,0]]
        save_image_ori = gt_image_np.astype(np.uint8)[:, :, [2,1,0]]
        out.write(save_image)
        out_rec.write(save_image_rec)
        out_ori.write(save_image_ori)
    
    # print("Time for test avg: ", (end_time-begin_time)/2000)
    # print("mlp time: ", mlp_time/2000, "gaussian time: ", gaussian_time/2000)  
    
    # 分区域保存
    # for region in region_mask.keys():
    #     out_list[region].release()
    
    avg_time = total_time/total_num
    print("avg time: ", avg_time, "avg fps = ", 1/avg_time)
    out.release()
    out_ori.release()
    out_rec.release()
    
def testing_compressed_model(args, lpt, opt, ppt, quantize):
    batch_size = 1
    set_random_seed(args.seed)

    ## deform model
    DeformModel = Deform_Model_smirk(args.device,
                                     exp_uesd_dim=args.exp_dim, mlp_layer=args.mlp_layer,hidden_dim=args.hidden_dim).to(args.device)
    
    DeformModel.training_setup()
    DeformModel.eval()

    ## dataloader
    data_dir = os.path.join('/home/xylem/IBC24/FlashAvatar-code/dataset', args.idname)
    mica_datadir = os.path.join('/home/xylem/IBC24/FlashAvatar-code/metrical-tracker/output', args.idname)
    save_dir = os.path.join('dataset', args.idname)
    logdir = os.path.join(save_dir, args.logname)
    
     # train process
    first_iter = 0
    
    quantize.use_shift = [bool(el) for el in quantize.use_shift]
    quantize.use_gumbel = [bool(el) for el in quantize.use_gumbel]
    quantize.gumbel_period = [bool(el) for el in quantize.gumbel_period]
    
    gaussians = GaussianModelSQ(args.sh, quantize)
    
    gaussians.training_setup(opt)
    
    
    if args.iteration:
        mlp_path = logdir + "/ckpt/chkpnt" + str(args.iteration) + "_mlp_best_fp16.pth"
        gaussian_path = logdir + "/ckpt/point_cloud_compressed" + str(args.iteration) + "_best_fp16.pkl"
        
        # 测试一下fp16的高斯参数和mlp参数
        # gaussian_path = "/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/Obama_smirk/log_smirk_eagles_150000/ckpt/separate/150000.pth"
        # mlp_path = "/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/Obama_smirk/log_smirk_eagles_150000/ckpt/fp16/mlp_150000"
        mlp_checkpoint = torch.load(mlp_path)
        if len(mlp_checkpoint) == 3:  # 新格式包含best_psnr
            (model_params, first_iter, best_psnr) = mlp_checkpoint
        elif len(mlp_checkpoint) == 2:  # 旧格式不包含best_psnr
            (model_params, first_iter) = mlp_checkpoint
            best_psnr = 0.0  # 使用默认值
        else:
            model_params = mlp_checkpoint
            
        # DeformModel.restore(model_params)
        DeformModel.restore_light(model_params)
        # gaussians.load_compressed_pkl(gaussian_path)
        gaussians.load_compressed_pkl_bit10(gaussian_path)
    
    gaussians.decode_latents()
    
    # 根据高斯区域来渲染不同区域的人脸
    region_mask_path = "/home/ljl/workspace/IBC24/FlashAvatar-code/flame/FLAME_masks/region_point_simple.json"
    region_mask = json.load(open(region_mask_path, "r"))
    # gaussian_idx_right_eyeball = region_mask["right_eyeball"]
    # print(gaussian_idx_right_eyeball)

    scene = Scene_mica_smirk(data_dir, mica_datadir, train_type=1, white_background=lpt.white_background,id_name=args.idname, quan_bits=args.quan_bit, device = args.device)
    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)
    
    
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # 对每隔区域的高斯都各自保存成不同的视频
    # out_list = {}
    # for region in region_mask.keys():
    #     vid_save_path = os.path.join(logdir, f'{region}.avi')
    #     out = cv2.VideoWriter(vid_save_path, fourcc, 30, (args.image_res*2, args.image_res), True)
    #     out_list[region] = out
        
    # 原本的保存
    vid_save_path = os.path.join(logdir, 'test.avi') 
    out = cv2.VideoWriter(vid_save_path, fourcc, 30, (args.image_res*2, args.image_res), True)
    
    # vid_save_path_rec = os.path.join(logdir, f'rec_{args.quan_bit}bit_exp_dim{args.exp_dim}_{args.chkpnt_number}_frame1000.avi')
    vid_save_path_rec = os.path.join(logdir, f'rec_{args.quan_bit}bit_exp_dim{args.exp_dim}_{args.chkpnt_number}_frame500_best_compressed.avi')
    
    out_rec = cv2.VideoWriter(vid_save_path_rec, fourcc, 30, (args.image_res, args.image_res), True)
    
    vid_save_path_ori = os.path.join(logdir, 'ori_frame500.avi')
    out_ori = cv2.VideoWriter(vid_save_path_ori, fourcc, 30, (args.image_res, args.image_res), True)
    
    viewpoint = scene.getCameras().copy()
    codedict = {}
    codedict['shape'] = scene.shape_param.to(args.device)
    DeformModel.example_init(codedict)
    
    # gaussian_time = 0.
    # mlp_time = 0.
    for iteration in range(len(viewpoint)):
        # mlp_begin_time = time.time()
        viewpoint_cam = viewpoint[iteration]
        frame_id = viewpoint_cam.uid

        # # deform gaussians
        codedict['expr'] = viewpoint_cam.exp_param
        codedict['pose'] = viewpoint_cam.pose_param
        codedict['eyelids'] = viewpoint_cam.eyelids
        codedict['jaw_pose'] = viewpoint_cam.jaw_pose
        
        
        # verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)
        verts_final, rot_delta, scale_coef = DeformModel.smirk_decode(codedict)
        gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])
        # mlp_time+=time.time()-mlp_begin_time
        # Render
        # gaussian_begin_time = time.time()
        
        # 对每个区域的高斯都保存成视频
        # for region in region_mask.keys():
        #     render_pkg = render_eagles(viewpoint_cam, gaussians, ppt, background, gaussian_region = region_mask[region])
        #     image= render_pkg["render"]
        #     image = image.clamp(0, 1)

        #     gt_image = viewpoint_cam.original_image
        #     save_image = np.zeros((args.image_res, args.image_res*2, 3))
        #     gt_image_np = (gt_image*255.).permute(1,2,0).detach().cpu().numpy()
        #     image_np = (image*255.).permute(1,2,0).detach().cpu().numpy()

        #     save_image[:, :args.image_res, :] = gt_image_np
        #     save_image[:, args.image_res:, :] = image_np
        #     save_image = save_image.astype(np.uint8)
        #     save_image = save_image[:,:,[2,1,0]]
            
        #     save_image_rec = image_np.astype(np.uint8)[:, :, [2,1,0]]
        #     save_image_ori = gt_image_np.astype(np.uint8)[:, :, [2,1,0]]
            
        #     out_list[region].write(save_image)
            
            
        render_pkg = render_eagles(viewpoint_cam, gaussians, ppt, background)
        image= render_pkg["render"]
        image = image.clamp(0, 1)

        gt_image = viewpoint_cam.original_image
        save_image = np.zeros((args.image_res, args.image_res*2, 3))
        gt_image_np = (gt_image*255.).permute(1,2,0).detach().cpu().numpy()
        image_np = (image*255.).permute(1,2,0).detach().cpu().numpy()

        save_image[:, :args.image_res, :] = gt_image_np
        save_image[:, args.image_res:, :] = image_np
        save_image = save_image.astype(np.uint8)
        save_image = save_image[:,:,[2,1,0]]
        
        save_image_rec = image_np.astype(np.uint8)[:, :, [2,1,0]]
        save_image_ori = gt_image_np.astype(np.uint8)[:, :, [2,1,0]]
        out.write(save_image)
        out_rec.write(save_image_rec)
        out_ori.write(save_image_ori)
    
    # print("Time for test avg: ", (end_time-begin_time)/2000)
    # print("mlp time: ", mlp_time/2000, "gaussian time: ", gaussian_time/2000)  
    
    # 分区域保存
    # for region in region_mask.keys():
    #     out_list[region].release()
    
    out.release()
    out_ori.release()
    out_rec.release()
    
    
def testing_compress(args, lpt, opt, ppt, quantize):
    batch_size = 1
    set_random_seed(args.seed)

    DeformModel = Deform_Model_smirk(args.device,
                                     exp_uesd_dim=args.exp_dim, mlp_layer=args.mlp_layer).to(args.device)
    DeformModel.training_setup()
    DeformModel.eval()
    ## dataloader
    data_dir = os.path.join('/home/xylem/IBC24/FlashAvatar-code/dataset', args.idname)
    mica_datadir = os.path.join('/home/xylem/IBC24/FlashAvatar-code/metrical-tracker/output', args.idname)
    save_dir = os.path.join('dataset', args.idname)
    logdir = os.path.join(save_dir, args.logname)
    
    quantize.use_shift = [bool(el) for el in quantize.use_shift]
    quantize.use_gumbel = [bool(el) for el in quantize.use_gumbel]
    quantize.gumbel_period = [bool(el) for el in quantize.gumbel_period]
    
    gaussians = GaussianModelSQ(args.sh, quantize)
    
    gaussians.training_setup(opt)
    
    
    if args.iteration:
        mlp_path = logdir + "/ckpt/chkpnt" + str(args.iteration) + "_mlp.pth"
        gaussian_path = logdir + "/ckpt/point_cloud_compressed" + str(args.iteration) + ".pkl"
        
        mlp_checkpoint = torch.load(mlp_path)
        if len(mlp_checkpoint) == 3:  # 新格式包含best_psnr
            (model_params, first_iter, best_psnr) = mlp_checkpoint
        else:  # 旧格式不包含best_psnr
            (model_params, first_iter) = mlp_checkpoint
            best_psnr = 0.0  # 使用默认值
            
        DeformModel.restore(model_params)
        gaussians.load_compressed_pkl(gaussian_path)
    
    # save_path = "/home/ljl/workspace/IBC24/FlashAvatar-code/gaussian_attr_test/"+args.logname+"/"+args.iteration+ "/"
    # gaussians.plot_latents(save_path)
    
        
    mlp_path_fp16 = logdir + "/ckpt/chkpnt" + str(args.iteration) + "_mlp_best_fp16.pth"
    gaussian_path_fp16 = logdir + "/ckpt/point_cloud_compressed" + str(args.iteration) + "_best_fp16.pkl"
    # save_pth = os.path.join(logdir, 'ckpt', 'separate')+f"/{args.iteration}.pth"
    # # os.makedirs(save_pth, exist_ok=True)
    
    # 保存各个属性
    gaussians.save_compressed_pkl_light_bit10(gaussian_path_fp16, quantize)
    
    # save_pth = os.path.join(logdir, 'ckpt', 'compressed')+f"/{args.iteration}.pth"
    # gaussians.save_compressed_pkl_light(save_pth, quantize)
    
    # # 保存fp16版本高斯属性
    # save_pth = os.path.join(logdir, 'ckpt', 'fp16')+f"/gaussians_{args.iteration}.pth"
    # gaussians.save_compressed_pkl_light_fp16(save_pth, quantize)
    
    # 保存fp16版本mlp
    # save_pth = os.path.join(logdir, 'ckpt', 'fp16')+f"/mlp_{args.iteration}.pth"
    # os.makedirs(save_pth, exist_ok=True)
    # DeformModel.capture_light()
    torch.save(DeformModel.capture_light(), mlp_path_fp16)
    
    
    # gaussians.decode_latents()

            
if __name__ == "__main__":
    
    # Config file is used for argument defaults. Command line arguments override config file.
    config_path = sys.argv[sys.argv.index("--config")+1] if "--config" in sys.argv else None
    if config_path:
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = {}
    config = defaultdict(lambda: {}, config)


    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser, config['opt_params'])
    pp = PipelineParams(parser)
    qp = QuantizeParams(parser, config['quantize_params'])
    
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--idname', type=str, default='id1_25', help='id name')
    parser.add_argument('--logname', type=str, default='log', help='log name')
    parser.add_argument('--exp_dim', type=int, default=50, help='exp used dim.')
    parser.add_argument('--image_res', type=int, default=512, help='image resolution')
    parser.add_argument('--quan_bit', type=int, default=0, help='')
    parser.add_argument('--sh', type=int, default=3, help='')
    parser.add_argument('--smooth', type=bool, default=False, help='')
    parser.add_argument("--start_checkpoint", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default = "/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/Obama_smirk/log_smirk_eagles_150000/ckpt/chkpnt150000_mlp.pth")
    parser.add_argument("--iteration", type= str , default = "150000")
    parser.add_argument("--mlp_layer", type= int , default = 6)
    parser.add_argument("--hidden_dim", type= int , default = 256)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--compress_test", action="store_true") 
    parser.add_argument("--chkpnt_number", type= int , default = 150000)
    parser.add_argument("--loss_type", type=str, default='lpips', choices=['lpips', 'ssim'],
                       help='Type of perceptual loss to use: lpips or ssim')
    
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"
    lp_args = lp.extract(args)
    op_args = op.extract(args)
    pp_args = pp.extract(args)
    qp_args = qp.extract(args)
    
    if args.compress_test:
        testing_compress(args, lp_args, op_args, pp_args, qp_args)
    else:
        if not args.skip_train:
            net_train_time = training(args, lp_args, op_args, pp_args, qp_args)
    
        if not args.skip_test:
            testing(args, lp_args, op_args, pp_args, qp_args)
            # testing_compressed_model(args, lp_args, op_args, pp_args, qp_args)
        
    
    
    