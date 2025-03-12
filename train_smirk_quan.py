import os, sys 
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
import cv2
import lpips

from scene import GaussianModel, Scene_mica,Scene_mica_smirk
from src.deform_model import Deform_Model,Deform_Model_smirk
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams_Ori
from utils.loss_utils import huber_loss
from utils.general_utils import normalize_for_percep


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

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams_Ori(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--idname', type=str, default='id1_25', help='id name')
    parser.add_argument('--logname', type=str, default='log', help='log name')
    parser.add_argument('--exp_dim', type=int, default=50, help='exp used dim.')
    parser.add_argument('--image_res', type=int, default=512, help='image resolution')
    parser.add_argument('--quan_bit', type=int, default=0, help='')
    parser.add_argument('--sh', type=int, default=3, help='')
    parser.add_argument('--smooth', type=bool, default=False, help='')
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"
    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)

    batch_size = 1
    set_random_seed(args.seed)

    percep_module = lpips.LPIPS(net='vgg').to(args.device)

    print("id_name:",args.idname, "exp_dim: ", args.exp_dim,"use_quan: ", args.quan_bit)
    ## deform model
    DeformModel = Deform_Model_smirk(args.device,
                                     exp_uesd_dim=args.exp_dim).to(args.device)
    
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
    scene = Scene_mica_smirk(data_dir, mica_datadir, train_type=0, white_background=lpt.white_background, device = args.device,id_name=args.idname, quan_bits=args.quan_bit)
    
    first_iter = 0
    # gaussians = GaussianModel(lpt.sh_degree)
    gaussians = GaussianModel(args.sh)
    gaussians.training_setup(opt)
    if args.start_checkpoint:
        (model_params, gauss_params, first_iter) = torch.load(args.start_checkpoint)
        DeformModel.restore(model_params)
        gaussians.restore(gauss_params, opt)

    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)
    
    codedict = {}
    codedict['shape'] = scene.shape_param.to(args.device)
    DeformModel.example_init(codedict)

    viewpoint_stack = None
    first_iter += 1
    mid_num = 15000
    total_iterations = 200000
    # for iteration in range(first_iter, opt.iterations + 1):
    for iteration in range(first_iter, total_iterations + 1):
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
        frame_id = viewpoint_cam.uid

        # deform gaussians
        
        # # old deform
        # codedict['expr'] = viewpoint_cam.exp_param
        # codedict['eyes_pose'] = viewpoint_cam.eyes_pose
        # codedict['eyelids'] = viewpoint_cam.eyelids
        # codedict['jaw_pose'] = viewpoint_cam.jaw_pose 
        # 不要detail
        # codedict['detail'] = viewpoint_cam.detail 
        
        # #emoca deform
        # codedict['expr'] = viewpoint_cam.exp_param
        # codedict['pose'] = viewpoint_cam.pose_param
        
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
        render_pkg = render(viewpoint_cam, gaussians, ppt, background)
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image
        mouth_mask = viewpoint_cam.mouth_mask
        
        loss_huber = huber_loss(image, gt_image, 0.1) + 40*huber_loss(image*mouth_mask, gt_image*mouth_mask, 0.1)
        
        loss_G = 0.
        head_mask = viewpoint_cam.head_mask
        image_percep = normalize_for_percep(image*head_mask)
        gt_image_percep = normalize_for_percep(gt_image*head_mask)
        if iteration>mid_num:
            loss_G = torch.mean(percep_module.forward(image_percep, gt_image_percep))*0.05
            
        if args.smooth:
            # 加上平滑损失，用高斯的feature_rest的方差表示平滑程度
            feature_rest = gaussians.get_features_rest
            # print(feature_rest.shape)
            # 展平为每个数据点的所有维度
            feature_rest = feature_rest.view(feature_rest.shape[0], -1)  # shape: (14000, 45)

            # 计算每个维度上的方差
            variance_per_dimension = torch.var(feature_rest, dim=0, unbiased=True)  # shape: (45,)

            # 将各个维度的方差相加
            loss_smooth = torch.sum(variance_per_dimension)
            # print(loss_smooth)

            
            loss = loss_huber*1 + loss_G*1 + loss_smooth*0.01
        
        else:
            loss = loss_huber*1 + loss_G*1
        

        loss.backward()

        with torch.no_grad():
            # Optimizer step
            # if iteration < opt.iterations :
            if iteration < total_iterations :
                gaussians.optimizer.step()
                DeformModel.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                DeformModel.optimizer.zero_grad(set_to_none = True)
            
            # print loss
            if iteration % 500 == 0:
                if iteration<=mid_num:
                    print("step: %d, huber: %.5f" %(iteration, loss_huber.item()))
                else:
                    if args.smooth:
                        print("step: %d, huber: %.5f, percep: %.5f, smooth: %.5f" %(iteration, loss_huber.item(), loss_G.item(), loss_smooth.item()))
                    else:
                        print("step: %d, huber: %.5f, percep: %.5f" %(iteration, loss_huber.item(), loss_G.item()))
            
            # visualize results
            if iteration % 500 == 0 or iteration==1:
                save_image = np.zeros((args.image_res, args.image_res*2, 3))
                gt_image_np = (gt_image*255.).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                image = image.clamp(0, 1)
                image_np = (image*255.).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                save_image[:, :args.image_res, :] = gt_image_np
                save_image[:, args.image_res:, :] = image_np
                cv2.imwrite(os.path.join(train_dir, f"{iteration}.jpg"), save_image[:,:,[2,1,0]])
            
            # save checkpoint
            if iteration % 5000 == 0:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((DeformModel.capture(), gaussians.capture(), iteration), model_dir + "/chkpnt" + str(iteration) + ".pth")

           