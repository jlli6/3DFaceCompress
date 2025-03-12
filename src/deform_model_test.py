import sys
import pickle
import torch
import numpy as np
from torch import nn
import math
import torch.nn.functional as F
from pytorch3d.io import load_obj

from flame import FLAME_mica, parse_args, FLAME
from utils.general_utils import Pytorch3dRasterizer, Embedder, load_binary_pickle, a_in_b_torch, face_vertices_gen
from skimage.io import imsave
from pathlib import Path
# emoca 
from gdl.models.DECA import DecaModule
from gdl_apps.EMOCA.utils.load import load_model

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, axis_angle_to_matrix


# def load_deca():
#     checkpoint = "/home/xylem/IBC24/emoca/assets/EMOCA/models/EMOCA_v2_lr_mse_20/detail/checkpoints/deca-epoch=10-val_loss/dataloader_idx_0=3.25521111.ckpt"
#     checkpoint_kwargs = {
#         "model_params": cfg.model,
#         "learning_params": cfg.learning,
#         "inout_params": cfg.inout,
#         "stage_name": "testing",
#     }
#     deca = DecaModule.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **checkpoint_kwargs)

def _fix_image( image):
    if image.max() < 30.:
        image = image * 255.
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def torch_img_to_np(img):
    return img.detach().cpu().numpy().transpose(1, 2, 0)

class Deform_Model(nn.Module):
    def __init__(self, device, exp_uesd_dim= 100 ):
        super().__init__()
        self.device = device
        
        mica_flame_config = parse_args()
        self.flame_model = FLAME_mica(mica_flame_config).to(self.device)
        # self.flame_model = FLAME(mica_flame_config).to(self.device)
        
        ## old
        self.default_shape_code = torch.zeros(1, 300, device=self.device)
        self.default_expr_code = torch.zeros(1, 100, device=self.device)
        # self.default_expr_code = torch.zeros(1, 50, device=self.device)
        self.exp_uesd_dim = exp_uesd_dim
        # emoca
        # self.default_shape_code = torch.zeros(1, 100, device=self.device)
        # self.default_expr_code = torch.zeros(1, 50, device=self.device)
        
        # ## smirk
        # self.default_shape_code = torch.zeros(1, 300, device=self.device)
        # self.default_expr_code = torch.zeros(1, 50, device=self.device)
        # self.default_pose_code = torch.zeros(1, 3, device=self.device)
        # self.default_jaw_code = torch.zeros(1, 3, device=self.device)
        
        # positional encoding
        self.pts_freq = 8
        self.pts_embedder = Embedder(self.pts_freq)
        
        _, faces, aux = load_obj('flame/FlameMesh.obj', load_textures=False)
        uv_coords = aux.verts_uvs[None, ...]
        uv_coords = uv_coords * 2 - 1
        uv_coords[..., 1] = - uv_coords[..., 1]
        self.uvcoords = torch.cat([uv_coords, uv_coords[:, :, 0:1] * 0. + 1.], -1).to(self.device)
        self.uvfaces = faces.textures_idx[None, ...].to(self.device)
        self.tri_faces = faces.verts_idx[None, ...].to(self.device)
        
        # rasterizer
        self.uv_size = 128
        self.uv_rasterizer = Pytorch3dRasterizer(self.uv_size)
        
        # flame mask
        flame_mask_path = "flame/FLAME_masks/FLAME_masks.pkl"   
        flame_mask_dic = load_binary_pickle(flame_mask_path) 
        boundary_id = flame_mask_dic['boundary']
        full_id = np.array(range(5023)).astype(int)
        neckhead_id_list = list(set(full_id)-set(boundary_id))
        self.neckhead_id_list = neckhead_id_list
        self.neckhead_id_tensor = torch.tensor(self.neckhead_id_list, dtype=torch.int64).to(self.device)
        self.init_networks()

    def init_networks(self):       
        ## full mica 
        self.deformNet = MLP(
            input_dim=self.pts_embedder.dim_embeded + 120, # 120, 184,56, 58
            output_dim=10,
            hidden_dim=256,
            hidden_layers=6
        )
        
    def example_init(self, codedict):
        # speed up
        shape_code = codedict['shape'].detach()
        batch_size = shape_code.shape[0]
        
        # mica flame
        geometry_shape = self.flame_model.forward_geo(
            shape_code,
            expression_params = self.default_expr_code
        )
        
        # flame
        # param_dictionary = {}
        # param_dictionary['shape_params'] = shape_code
        # param_dictionary['expression_params'] = self.default_expr_code
        # param_dictionary['pose_params'] = self.default_pose_code
        # param_dictionary['jaw_params'] = self.default_jaw_code
        
        # geometry_shape = self.flame_model.forward(
        #     param_dictionary = param_dictionary
        # )

        face_vertices_shape = face_vertices_gen(geometry_shape, self.tri_faces.expand(batch_size, -1, -1))
        rast_out, pix_to_face, bary_coords = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1),
                                         self.uvfaces.expand(batch_size, -1, -1),
                                         face_vertices_shape)
        self.pix_to_face_ori = pix_to_face
        self.bary_coords = bary_coords

        uvmask = rast_out[:, -1].unsqueeze(1)
        uvmask_flaten = uvmask[0].view(uvmask.shape[1], -1).permute(1, 0).squeeze(1) # batch=1
        self.uvmask_flaten_idx = (uvmask_flaten[:]>0)

        pix_to_face_flaten = pix_to_face[0].clone().view(-1) # batch=1
        self.pix_to_face = pix_to_face_flaten[self.uvmask_flaten_idx] # pix to face idx
        self.pix_to_v_idx = self.tri_faces[0, self.pix_to_face, :] # pix to vert idx

        uv_vertices_shape = rast_out[:, :3]
        uv_vertices_shape_flaten = uv_vertices_shape[0].view(uv_vertices_shape.shape[1], -1).permute(1, 0) # batch=1       
        uv_vertices_shape = uv_vertices_shape_flaten[self.uvmask_flaten_idx].unsqueeze(0)

        self.uv_vertices_shape = uv_vertices_shape # for cano init
        self.uv_vertices_shape_embeded = self.pts_embedder(uv_vertices_shape)
        # print("shape of self.uv_vertices_shape:",self.uv_vertices_shape.shape)
        # print("shape of self.uv_vertices_shape_embeded:",self.uv_vertices_shape_embeded.shape)
        self.v_num = self.uv_vertices_shape_embeded.shape[1]

        # mask
        self.uv_head_idx = (
            a_in_b_torch(self.pix_to_v_idx[:,0], self.neckhead_id_tensor)
            & a_in_b_torch(self.pix_to_v_idx[:,1], self.neckhead_id_tensor)
            & a_in_b_torch(self.pix_to_v_idx[:,2], self.neckhead_id_tensor)
        )
    
    def decode(self, codedict):
        shape_code = codedict['shape'].detach()
        expr_code = codedict['expr'].detach()
        
        expr_code[0,self.exp_uesd_dim:] = 0
        jaw_pose = codedict['jaw_pose'].detach()
        eyelids = codedict['eyelids'].detach()
        eyes_pose = codedict['eyes_pose'].detach()
        batch_size = shape_code.shape[0]
        
        
        # print("shape_code的维度：",shape_code.shape,shape_code.shape[0],shape_code.shape[1])
        condition = torch.cat((expr_code, jaw_pose, eyes_pose, eyelids), dim=1)

        # MLP
        condition = condition.unsqueeze(1).repeat(1, self.v_num, 1)
        # 输出一下各个向量的维度
        # print("shape_code的维度：",shape_code.shape)
        # print("expr_code的维度：",expr_code.shape)
        # print("jaw_pose的维度：",jaw_pose.shape)
        # print("eyelids的维度：",eyelids.shape)
        # print("eyes_pose的维度：",eyes_pose.shape)
        # print("condition的维度：",condition.shape)
        # print("self.uv_vertices_shape_embeded的维度：",self.uv_vertices_shape_embeded.shape)
        # print("self.v_num：",self.v_num)
        uv_vertices_shape_embeded_condition = torch.cat((self.uv_vertices_shape_embeded, condition), dim=2)
        deforms = self.deformNet(uv_vertices_shape_embeded_condition)
        # print("输入的维度：",uv_vertices_shape_embeded_condition.shape)
        deforms = torch.tanh(deforms)
        # print("输出的维度：",deforms.shape)
        uv_vertices_deforms = deforms[..., :3]
        rot_delta_0 = deforms[..., 3:7]
        rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
        rot_delta_v = rot_delta_0[..., 1:]
        rot_delta = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
        scale_coef = deforms[..., 7:]
        scale_coef = torch.exp(scale_coef)

        geometry = self.flame_model.forward_geo(
            shape_code,
            expression_params=expr_code,
            jaw_pose_params=jaw_pose,
            eye_pose_params=eyes_pose,
            eyelid_params=eyelids,
        )
        face_vertices = face_vertices_gen(geometry, self.tri_faces.expand(batch_size, -1, -1))

        # rasterize face_vertices to uv space
        D = face_vertices.shape[-1] # 3
        attributes = face_vertices.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = self.bary_coords.shape
        idx = self.pix_to_face_ori.clone().view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (self.bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        uv_vertices = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        uv_vertices_flaten = uv_vertices[0].view(uv_vertices.shape[1], -1).permute(1, 0) # batch=1
        uv_vertices = uv_vertices_flaten[self.uvmask_flaten_idx].unsqueeze(0)

        verts_final = uv_vertices + uv_vertices_deforms

        # conduct mask
        verts_final = verts_final[:, self.uv_head_idx, :]
        rot_delta = rot_delta[:, self.uv_head_idx, :]
        scale_coef = scale_coef[:, self.uv_head_idx, :]

        return verts_final, rot_delta, scale_coef
    
    
    def smirk_decode(self, codedict):
        shape_code = codedict['shape'].detach()
        expr_code = codedict['expr'].detach()
        jaw_pose = codedict['jaw_pose'].detach()
        eyelids = codedict['eyelids'].detach()
        pose = codedict['pose'].detach()
        batch_size = shape_code.shape[0]
        
        # print("shape_code的维度：",shape_code.shape,shape_code.shape[0],shape_code.shape[1])
        condition = torch.cat((expr_code, jaw_pose, pose, eyelids), dim=1)

        # MLP
        condition = condition.unsqueeze(1).repeat(1, self.v_num, 1)
        
        uv_vertices_shape_embeded_condition = torch.cat((self.uv_vertices_shape_embeded, condition), dim=2)
        deforms = self.deformNet(uv_vertices_shape_embeded_condition)
        # print("输入的维度：",uv_vertices_shape_embeded_condition.shape)
        deforms = torch.tanh(deforms)
        # print("输出的维度：",deforms.shape)
        uv_vertices_deforms = deforms[..., :3]
        # print("uv_vertices_deforms的维度：",uv_vertices_deforms.shape)
        rot_delta_0 = deforms[..., 3:7]
        rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
        rot_delta_v = rot_delta_0[..., 1:]
        rot_delta = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
        scale_coef = deforms[..., 7:]
        scale_coef = torch.exp(scale_coef)

        # param_dict = {}
        # param_dict['shape_params'] = shape_code
        # param_dict['expression_params'] = expr_code
        # param_dict['jaw_params'] = jaw_pose
        # param_dict['pose_params'] = pose
        # param_dict['eyelid_params'] = eyelids
        
        # geometry = self.flame_model.forward(
        #     param_dict
        # )
        
        geometry = self.flame_model.forward_geo(
            shape_code,
            expression_params=expr_code,
            jaw_pose_params=jaw_pose,
            rot_params=pose,
            eyelid_params=eyelids,
        )
        face_vertices = face_vertices_gen(geometry, self.tri_faces.expand(batch_size, -1, -1))

        # rasterize face_vertices to uv space
        D = face_vertices.shape[-1] # 3
        attributes = face_vertices.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = self.bary_coords.shape
        idx = self.pix_to_face_ori.clone().view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (self.bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        uv_vertices = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        uv_vertices_flaten = uv_vertices[0].view(uv_vertices.shape[1], -1).permute(1, 0) # batch=1
        uv_vertices = uv_vertices_flaten[self.uvmask_flaten_idx].unsqueeze(0)

        verts_final = uv_vertices + uv_vertices_deforms

        # conduct mask
        verts_final = verts_final[:, self.uv_head_idx, :]
        rot_delta = rot_delta[:, self.uv_head_idx, :]
        scale_coef = scale_coef[:, self.uv_head_idx, :]

        return verts_final, rot_delta, scale_coef
    
    def emoca_decode(self, codedict):
        shape_code = codedict['shape'].detach()
        expr_code = codedict['expr'].detach()
        pose_code = codedict['pose'].detach()
        detail = codedict['detail'].detach()
        # eyes_pose = codedict['eyes_pose'].detach()
        batch_size = shape_code.shape[0]
        # print("shape_code的维度：",shape_code.shape,shape_code.shape[0],shape_code.shape[1])
        
        # condition = torch.cat((expr_code, pose_code, detail), dim=1)
        condition = torch.cat((expr_code, pose_code), dim=1) # 去掉detail

        # MLP
        condition = condition.unsqueeze(1).repeat(1, self.v_num, 1)
        # 输出一下各个向量的维度
        # print("shape_code的维度：",shape_code.shape)
        # print("expr_code的维度：",expr_code.shape)
        # print("jaw_pose的维度：",jaw_pose.shape)
        # print("eyelids的维度：",eyelids.shape)
        # print("eyes_pose的维度：",eyes_pose.shape)
        # print("condition的维度：",condition.shape)
        # print("self.uv_vertices_shape_embeded的维度：",self.uv_vertices_shape_embeded.shape)
        # print("self.v_num：",self.v_num)
        uv_vertices_shape_embeded_condition = torch.cat((self.uv_vertices_shape_embeded, condition), dim=2)
        deforms = self.deformNet(uv_vertices_shape_embeded_condition)
        # print("输入的维度：",uv_vertices_shape_embeded_condition.shape)
        deforms = torch.tanh(deforms)
        # print("输出的维度：",deforms.shape)
        uv_vertices_deforms = deforms[..., :3]
        rot_delta_0 = deforms[..., 3:7]
        rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
        rot_delta_v = rot_delta_0[..., 1:]
        rot_delta = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
        scale_coef = deforms[..., 7:]
        scale_coef = torch.exp(scale_coef)

        geometry = self.flame_model.forward_geo(
            shape_code,
            expression_params=expr_code,
            # jaw_pose_params=jaw_pose,
            # eye_pose_params=eyes_pose,
            # eyelid_params=eyelids,
        )
        face_vertices = face_vertices_gen(geometry, self.tri_faces.expand(batch_size, -1, -1))

        # rasterize face_vertices to uv space
        D = face_vertices.shape[-1] # 3
        attributes = face_vertices.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = self.bary_coords.shape
        idx = self.pix_to_face_ori.clone().view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (self.bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        uv_vertices = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        uv_vertices_flaten = uv_vertices[0].view(uv_vertices.shape[1], -1).permute(1, 0) # batch=1
        uv_vertices = uv_vertices_flaten[self.uvmask_flaten_idx].unsqueeze(0)

        verts_final = uv_vertices + uv_vertices_deforms

        # conduct mask
        verts_final = verts_final[:, self.uv_head_idx, :]
        rot_delta = rot_delta[:, self.uv_head_idx, :]
        scale_coef = scale_coef[:, self.uv_head_idx, :]

        return verts_final, rot_delta, scale_coef
    
    
    def capture(self):
        return (
            self.deformNet.state_dict(),
            self.optimizer.state_dict(),
        )
    
    def restore(self, model_args):
        (net_dict,
         opt_dict) = model_args
        self.deformNet.load_state_dict(net_dict)
        self.training_setup()
        self.optimizer.load_state_dict(opt_dict)

    
    def training_setup(self):
        params_group = [
            {'params': self.deformNet.parameters(), 'lr': 1e-4},
        ]
        self.optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999))

    
    def get_template(self):
        geometry_template = self.flame_model.forward_geo(
            self.default_shape_code,
            self.default_expr_code,
            self.default_jaw_pose,
            eye_pose_params=self.default_eyes_pose,
        )

        return geometry_template
    
class Deform_Model_smirk(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        mica_flame_config = parse_args()
        
        mica_flame_config.num_exp_params = 50
        mica_flame_config.num_shape_params == 300
        self.flame_model = FLAME_mica(mica_flame_config).to(self.device)
        # self.flame_model = FLAME(mica_flame_config).to(self.device)
        
        # ## old
        # self.default_shape_code = torch.zeros(1, 300, device=self.device)
        # self.default_expr_code = torch.zeros(1, 100, device=self.device)
        
        # emoca
        # self.default_shape_code = torch.zeros(1, 100, device=self.device)
        # self.default_expr_code = torch.zeros(1, 50, device=self.device)
        
        ## smirk
        self.default_shape_code = torch.zeros(1, 300, device=self.device)
        self.default_expr_code = torch.zeros(1, 50, device=self.device)
        self.default_pose_code = torch.zeros(1, 3, device=self.device)
        self.default_jaw_code = torch.zeros(1, 3, device=self.device)
        
        # positional encoding
        self.pts_freq = 8
        self.pts_embedder = Embedder(self.pts_freq)
        
        _, faces, aux = load_obj('flame/FlameMesh.obj', load_textures=False)
        uv_coords = aux.verts_uvs[None, ...]
        uv_coords = uv_coords * 2 - 1
        uv_coords[..., 1] = - uv_coords[..., 1]
        self.uvcoords = torch.cat([uv_coords, uv_coords[:, :, 0:1] * 0. + 1.], -1).to(self.device)
        self.uvfaces = faces.textures_idx[None, ...].to(self.device)
        self.tri_faces = faces.verts_idx[None, ...].to(self.device)
        
        # rasterizer
        self.uv_size = 128
        self.uv_rasterizer = Pytorch3dRasterizer(self.uv_size)
        
        # flame mask
        flame_mask_path = "flame/FLAME_masks/FLAME_masks.pkl"   
        flame_mask_dic = load_binary_pickle(flame_mask_path) 
        boundary_id = flame_mask_dic['boundary']
        full_id = np.array(range(5023)).astype(int)
        neckhead_id_list = list(set(full_id)-set(boundary_id))
        self.neckhead_id_list = neckhead_id_list
        self.neckhead_id_tensor = torch.tensor(self.neckhead_id_list, dtype=torch.int64).to(self.device)
        self.init_networks()

    def init_networks(self):       
        ## full mica 
        self.deformNet = MLP(
            input_dim=self.pts_embedder.dim_embeded + 58-3, # 120, 184,56, 58
            output_dim=10,
            hidden_dim=256,
            hidden_layers=6
        )
        
    def example_init(self, codedict):
        # speed up
        shape_code = codedict['shape'].detach()
        batch_size = shape_code.shape[0]
        
        # mica flame
        geometry_shape = self.flame_model.forward_geo(
            shape_code,
            expression_params = self.default_expr_code
        )
        
        # # flame
        # param_dictionary = {}
        # param_dictionary['shape_params'] = shape_code
        # param_dictionary['expression_params'] = self.default_expr_code
        # param_dictionary['pose_params'] = self.default_pose_code
        # param_dictionary['jaw_params'] = self.default_jaw_code
        
        # geometry_shape = self.flame_model.forward(
        #     param_dictionary = param_dictionary
        # )

        face_vertices_shape = face_vertices_gen(geometry_shape, self.tri_faces.expand(batch_size, -1, -1))
        rast_out, pix_to_face, bary_coords = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1),
                                         self.uvfaces.expand(batch_size, -1, -1),
                                         face_vertices_shape)
        self.pix_to_face_ori = pix_to_face
        self.bary_coords = bary_coords

        uvmask = rast_out[:, -1].unsqueeze(1)
        uvmask_flaten = uvmask[0].view(uvmask.shape[1], -1).permute(1, 0).squeeze(1) # batch=1
        self.uvmask_flaten_idx = (uvmask_flaten[:]>0)

        pix_to_face_flaten = pix_to_face[0].clone().view(-1) # batch=1
        self.pix_to_face = pix_to_face_flaten[self.uvmask_flaten_idx] # pix to face idx
        self.pix_to_v_idx = self.tri_faces[0, self.pix_to_face, :] # pix to vert idx

        uv_vertices_shape = rast_out[:, :3]
        uv_vertices_shape_flaten = uv_vertices_shape[0].view(uv_vertices_shape.shape[1], -1).permute(1, 0) # batch=1       
        uv_vertices_shape = uv_vertices_shape_flaten[self.uvmask_flaten_idx].unsqueeze(0)

        self.uv_vertices_shape = uv_vertices_shape # for cano init
        self.uv_vertices_shape_embeded = self.pts_embedder(uv_vertices_shape)
        # print("shape of self.uv_vertices_shape:",self.uv_vertices_shape.shape)
        # print("shape of self.uv_vertices_shape_embeded:",self.uv_vertices_shape_embeded.shape)
        self.v_num = self.uv_vertices_shape_embeded.shape[1]

        # mask
        self.uv_head_idx = (
            a_in_b_torch(self.pix_to_v_idx[:,0], self.neckhead_id_tensor)
            & a_in_b_torch(self.pix_to_v_idx[:,1], self.neckhead_id_tensor)
            & a_in_b_torch(self.pix_to_v_idx[:,2], self.neckhead_id_tensor)
        )
    
    def smirk_decode(self, codedict):
        shape_code = codedict['shape'].detach()
        expr_code = codedict['expr'].detach()
        jaw_pose = codedict['jaw_pose'].detach()
        eyelids = codedict['eyelids'].detach()
        pose = codedict['pose'].detach()
        batch_size = shape_code.shape[0]
        
        # print("shape_code的维度：",shape_code.shape,shape_code.shape[0],shape_code.shape[1])
        # condition = torch.cat((expr_code, jaw_pose, pose, eyelids), dim=1)
        condition = torch.cat((expr_code, jaw_pose, eyelids), dim=1)

        # MLP
        condition = condition.unsqueeze(1).repeat(1, self.v_num, 1)
        
        uv_vertices_shape_embeded_condition = torch.cat((self.uv_vertices_shape_embeded, condition), dim=2)
        deforms = self.deformNet(uv_vertices_shape_embeded_condition)
        # print("输入的维度：",uv_vertices_shape_embeded_condition.shape)
        deforms = torch.tanh(deforms)
        # print("输出的维度：",deforms.shape)
        uv_vertices_deforms = deforms[..., :3]
        # print("uv_vertices_deforms的维度：",uv_vertices_deforms.shape)
        rot_delta_0 = deforms[..., 3:7]
        rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
        rot_delta_v = rot_delta_0[..., 1:]
        rot_delta = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
        scale_coef = deforms[..., 7:]
        scale_coef = torch.exp(scale_coef)

        # param_dict = {}
        # param_dict['shape_params'] = shape_code
        # param_dict['expression_params'] = expr_code
        # param_dict['jaw_params'] = jaw_pose
        # param_dict['pose_params'] = pose
        # param_dict['eyelid_params'] = eyelids
        
        # geometry = self.flame_model.forward(
        #     param_dict
        # )
        
        # # 将轴角转为旋转矩阵
        # pose = axis_angle_to_matrix(pose)
        # pose = matrix_to_rotation_6d(pose)
        
        jaw_pose = axis_angle_to_matrix(jaw_pose)
        jaw_pose = matrix_to_rotation_6d(jaw_pose)
        
        
        geometry = self.flame_model.forward_geo(
            shape_code,
            expression_params=expr_code,
            jaw_pose_params=jaw_pose,
            # rot_params=pose,
            eyelid_params=eyelids,
        )
        face_vertices = face_vertices_gen(geometry, self.tri_faces.expand(batch_size, -1, -1))

        # rasterize face_vertices to uv space
        D = face_vertices.shape[-1] # 3
        attributes = face_vertices.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = self.bary_coords.shape
        idx = self.pix_to_face_ori.clone().view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (self.bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        
        
        uv_vertices = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        uv_vertices_flaten = uv_vertices[0].view(uv_vertices.shape[1], -1).permute(1, 0) # batch=1
        uv_vertices = uv_vertices_flaten[self.uvmask_flaten_idx].unsqueeze(0)

        verts_final = uv_vertices + uv_vertices_deforms

        # conduct mask
        verts_final = verts_final[:, self.uv_head_idx, :]
        rot_delta = rot_delta[:, self.uv_head_idx, :]
        scale_coef = scale_coef[:, self.uv_head_idx, :]

        return verts_final, rot_delta, scale_coef
    
    
    def capture(self):
        return (
            self.deformNet.state_dict(),
            self.optimizer.state_dict(),
        )
    
    def restore(self, model_args):
        (net_dict,
         opt_dict) = model_args
        self.deformNet.load_state_dict(net_dict)
        self.training_setup()
        self.optimizer.load_state_dict(opt_dict)

    
    def training_setup(self):
        params_group = [
            {'params': self.deformNet.parameters(), 'lr': 1e-4},
        ]
        self.optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999))

    
    def get_template(self):
        geometry_template = self.flame_model.forward_geo(
            self.default_shape_code,
            self.default_expr_code,
            self.default_jaw_pose,
            eye_pose_params=self.default_eyes_pose,
        )

        return geometry_template

class Deform_Model_emoca(nn.Module):
    def __init__(self, device, use_detail=False, use_pose = False, flame_type = "mica", exp_uesd_dim = 50, exp_dim_used_in50 = 50, mlp_mask = False):
        super().__init__()
        self.device = device
        self.use_detail = use_detail
        self.use_pose = use_pose
        self.flame_type = flame_type    
        self.exp_used_dim = exp_uesd_dim
        self.mlp_mask = mlp_mask
        self.exp_dim_used_in50 = exp_dim_used_in50
        self.level = 0
        mica_flame_config = parse_args()
        mica_flame_config.num_shape_params = 100
        mica_flame_config.num_exp_params = 50 # self.exp_used_dim
        
        
        if self.flame_type == "mica":
            self.flame_model = FLAME_mica(mica_flame_config).to(self.device)
            print("using mica flame")
        if self.flame_type == "emoca":
            print("using emoca flame")
            
        # self.flame_model = FLAME(mica_flame_config).to(self.device)
        
        ## old
        # self.default_shape_code = torch.zeros(1, 300, device=self.device)
        # self.default_expr_code = torch.zeros(1, 100, device=self.device)
        
        # emoca
        self.default_shape_code = torch.zeros(1, mica_flame_config.num_shape_params, device=self.device)
        self.default_expr_code = torch.zeros(1, mica_flame_config.num_exp_params, device=self.device)
        self.default_pose_code = torch.zeros(1, 6, device=self.device)
        # ## smirk
        # self.default_shape_code = torch.zeros(1, 300, device=self.device)
        # self.default_expr_code = torch.zeros(1, 50, device=self.device)
        # self.default_pose_code = torch.zeros(1, 3, device=self.device)
        # self.default_jaw_code = torch.zeros(1, 3, device=self.device)
        
        # positional encoding
        self.pts_freq = 8
        self.pts_embedder = Embedder(self.pts_freq)
        
        _, faces, aux = load_obj('flame/FlameMesh.obj', load_textures=False)
        uv_coords = aux.verts_uvs[None, ...]
        uv_coords = uv_coords * 2 - 1
        uv_coords[..., 1] = - uv_coords[..., 1]
        self.uvcoords = torch.cat([uv_coords, uv_coords[:, :, 0:1] * 0. + 1.], -1).to(self.device)
        self.uvfaces = faces.textures_idx[None, ...].to(self.device)
        self.tri_faces = faces.verts_idx[None, ...].to(self.device)
        
        # rasterizer
        self.uv_size = 128
        self.uv_rasterizer = Pytorch3dRasterizer(self.uv_size)
        
        # flame mask
        flame_mask_path = "flame/FLAME_masks/FLAME_masks.pkl"   
        flame_mask_dic = load_binary_pickle(flame_mask_path) 
        boundary_id = flame_mask_dic['boundary']
        full_id = np.array(range(5023)).astype(int)
        neckhead_id_list = list(set(full_id)-set(boundary_id))
        self.neckhead_id_list = neckhead_id_list
        self.neckhead_id_tensor = torch.tensor(self.neckhead_id_list, dtype=torch.int64).to(self.device)
        self.init_networks()
        
        # emoca 相关
        path_to_models = "/home/xylem/IBC24/emoca/assets/EMOCA/models"
        model_name = "EMOCA_v2_lr_mse_20"
        mode = "detail"
        self.emoca, conf = load_model(path_to_models, model_name, mode,50)
        self.emoca.cuda()
        self.emoca.eval()
        

    def init_networks(self):       
        ## full mica
        input_dim = 50 + 3 # exp + jaw
        if self.mlp_mask:
            input_dim = 50 + 3
        if self.use_pose:
            input_dim = input_dim +3
         
        self.deformNet = MLP(
            input_dim=self.pts_embedder.dim_embeded + input_dim, # 120, 184,56, 58
            output_dim=10,
            hidden_dim=256,
            hidden_layers=6
        )
        
    def example_init(self, codedict):
        # speed up
        shape_code = codedict['shape'].detach()
        batch_size = shape_code.shape[0]
        
        if self.flame_type == "mica":
        # mica flame
            geometry_shape = self.flame_model.forward_geo(
                shape_code,
                expression_params = self.default_expr_code
            )
        if self.flame_type == "emoca":
            trans_codedict = {}
            trans_codedict["shapecode"] = shape_code
            trans_codedict["expcode"] = self.default_expr_code
            trans_codedict["posecode"] = self.default_pose_code
            geometry_shape = self.emoca.get_flame_verts(trans_codedict)
        # flame
        # param_dictionary = {}
        # param_dictionary['shape_params'] = shape_code
        # param_dictionary['expression_params'] = self.default_expr_code
        # param_dictionary['pose_params'] = self.default_pose_code
        # param_dictionary['jaw_params'] = self.default_jaw_code
        
        # geometry_shape = self.flame_model.forward(
        #     param_dictionary = param_dictionary
        # )

        face_vertices_shape = face_vertices_gen(geometry_shape, self.tri_faces.expand(batch_size, -1, -1))
        rast_out, pix_to_face, bary_coords = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1),
                                         self.uvfaces.expand(batch_size, -1, -1),
                                         face_vertices_shape)
        self.pix_to_face_ori = pix_to_face
        self.bary_coords = bary_coords

        uvmask = rast_out[:, -1].unsqueeze(1)
        uvmask_flaten = uvmask[0].view(uvmask.shape[1], -1).permute(1, 0).squeeze(1) # batch=1
        self.uvmask_flaten_idx = (uvmask_flaten[:]>0)

        pix_to_face_flaten = pix_to_face[0].clone().view(-1) # batch=1
        self.pix_to_face = pix_to_face_flaten[self.uvmask_flaten_idx] # pix to face idx
        self.pix_to_v_idx = self.tri_faces[0, self.pix_to_face, :] # pix to vert idx

        uv_vertices_shape = rast_out[:, :3]
        uv_vertices_shape_flaten = uv_vertices_shape[0].view(uv_vertices_shape.shape[1], -1).permute(1, 0) # batch=1       
        uv_vertices_shape = uv_vertices_shape_flaten[self.uvmask_flaten_idx].unsqueeze(0)

        self.uv_vertices_shape = uv_vertices_shape # for cano init
        self.uv_vertices_shape_embeded = self.pts_embedder(uv_vertices_shape)
        print("shape of self.uv_vertices_shape:",self.uv_vertices_shape.shape)
        print("shape of self.uv_vertices_shape_embeded:",self.uv_vertices_shape_embeded.shape)
        self.v_num = self.uv_vertices_shape_embeded.shape[1]

        # mask
        self.uv_head_idx = (
            a_in_b_torch(self.pix_to_v_idx[:,0], self.neckhead_id_tensor)
            & a_in_b_torch(self.pix_to_v_idx[:,1], self.neckhead_id_tensor)
            & a_in_b_torch(self.pix_to_v_idx[:,2], self.neckhead_id_tensor)
        )
    
    def uv_downsample(sekf, input_tensor):

            # Step 1: 下采样到 128x128
            downsampled = F.interpolate(input_tensor, size=(128, 128), mode='bilinear', align_corners=False)  # [1, 1, 128, 128]

            # Step 2: 调整形状以匹配目标
            downsampled = downsampled.permute(0, 2, 3, 1)  # [1, 128, 128, 1]
            downsampled = downsampled.unsqueeze(-1)  # [1, 128, 128, 1, 1]
            downsampled = downsampled.expand(-1, -1, -1, -1, 3)  # [1, 128, 128, 1, 3]
            
            return downsampled
    
    def emoca_decode(self, codedict, iteration = 0):
        shape_code = codedict['shape'].detach()
        expr_code = codedict['expr'].detach()
        
        # expr code只取用到的维度
        # expr_code = expr_code[:,0:self.exp_used_dim]
        expr_code[:,self.exp_used_dim:] = 0
        
        if self.exp_dim_used_in50 != 50: # 仅仅在50维训练，少维度推理时用到
            expr_code[:,self.exp_dim_used_in50:] = 0
        
        # 如果mask,则随机将expr_code的后面10xn位 变为0，其中n= 0,1,2,3,4
        if self.mlp_mask:
            level = int((iteration-1)/50_000) + 1
            if self.level != level:
                print("mask level changed, from ",self.level, " to ",level)
                self.level = level
            n = torch.randint(0, level, (1,)).item()+1
            # mask_length = 50 - 10 * n  # 计算要设置为 0 的长度
            mask_length = 10 * n  # 计算要设置为 0 的长度
            expr_code[:, -mask_length:] = 0
            
        # simple mask
        # expr_code[:,self.exp_dim_used_in50:] = 0
        # if self.mlp_mask:
        #     mask_length = 10 * n  # 计算要设置为 0 的长度
        #     if mask_length > 0:  # 只有当 mask_length > 0 时需要处理
        #         expr_code[:, -mask_length:] = 0  # 将最后 mask_length 列设置为 0
            
        pose_code = codedict['pose'].detach()
        detail = codedict['detail'].detach()
        # cam = codedict['cam'].detach()
        # eyes_pose = codedict['eyes_pose'].detach()
        batch_size = shape_code.shape[0]
        # print("shape_code的维度：",shape_code.shape,shape_code.shape[0],shape_code.shape[1])
        if self.use_pose:
        # condition = torch.cat((shape_code,expr_code, pose_code), dim=1)
            condition = torch.cat((expr_code, pose_code), dim=1) # 去掉detail
        else:
            condition = torch.cat((expr_code, pose_code[:,3:]), dim=1) # only jaw pose


        # MLP
        condition = condition.unsqueeze(1).repeat(1, self.v_num, 1)
       
        uv_vertices_shape_embeded_condition = torch.cat((self.uv_vertices_shape_embeded, condition), dim=2)
        deforms = self.deformNet(uv_vertices_shape_embeded_condition)
        # print("输入的维度：",uv_vertices_shape_embeded_condition.shape)
        deforms = torch.tanh(deforms)
        # print("输出的维度：",deforms.shape)
        uv_vertices_deforms = deforms[..., :3]
        rot_delta_0 = deforms[..., 3:7]
        rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
        rot_delta_v = rot_delta_0[..., 1:]
        rot_delta = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
        scale_coef = deforms[..., 7:]
        scale_coef = torch.exp(scale_coef)
        
        if self.flame_type == "mica":
            if self.use_pose:
                geometry = self.flame_model.forward_geo(
                    shape_code,
                    expression_params=expr_code,
                    jaw_pose_params=pose_code[:,0:3],
                    rot_params=pose_code[:,3:],
                    # eyelid_params=detail,
                )
            else:
                jaw_pose_matrix = axis_angle_to_matrix(pose_code[:,3:])
                jaw_pose_code = matrix_to_rotation_6d(jaw_pose_matrix)
                geometry = self.flame_model.forward_geo(
                    shape_code,
                    expression_params=expr_code,
                    jaw_pose_params=jaw_pose_code,
                    # rot_params=pose_code[3:],
                    # eyelid_params=detail,
                )

        # geometry = self.flame_model.forward_geo(
        #     shape_code,
        #     expression_params=expr_code,
        #     # jaw_pose_params=jaw_pose,
        #     # eye_pose_params=eyes_pose,
        #     # eyelid_params=eyelids,
        # )
        trans_codedict = {}
        trans_codedict["shapecode"] = shape_code
        trans_codedict["expcode"] = expr_code
        trans_codedict["posecode"] = pose_code
        trans_codedict["detailcode"] = detail
        if self.use_pose:
                trans_codedict["posecode"] = pose_code
        else:
            trans_codedict["posecode"][:,0:3] = pose_code[:,0:3]*0
            
        if self.flame_type == "emoca":
            
            geometry = self.emoca.get_flame_verts(trans_codedict)
        
        
        
        
        face_vertices = face_vertices_gen(geometry, self.tri_faces.expand(batch_size, -1, -1))

        # rasterize face_vertices to uv space
        D = face_vertices.shape[-1] # 3
        attributes = face_vertices.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        
        N, H, W, K, _ = self.bary_coords.shape
        idx = self.pix_to_face_ori.clone().view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)    
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (self.bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        
        # 加上偏移
        if self.use_detail:
            uv_z, normal = self.emoca.deform_detail(geometry, trans_codedict)
            uv_z_downsample = self.uv_downsample(uv_z)
            face_vertices_normal = face_vertices_gen(normal, self.tri_faces.expand(batch_size, -1, -1))
            attributes_normal = face_vertices_normal.clone()
            attributes_normal = attributes_normal.view(attributes_normal.shape[0] * attributes_normal.shape[1], 3, attributes_normal.shape[-1])
            
            pixel_face_vals_normal = attributes_normal.gather(0, idx).view(N, H, W, K, 3, D)
            pixel_vals_normal = (self.bary_coords[..., None] * pixel_face_vals_normal).sum(dim=-2)
            pixel_vals_detail = pixel_vals + uv_z_downsample * pixel_vals_normal
            pixel_vals = pixel_vals_detail
        # pixel_vals = pixel_vals_detail
        
        uv_vertices = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        uv_vertices_flaten = uv_vertices[0].view(uv_vertices.shape[1], -1).permute(1, 0) # batch=1
        uv_vertices = uv_vertices_flaten[self.uvmask_flaten_idx].unsqueeze(0)

        verts_final = uv_vertices + uv_vertices_deforms

        # conduct mask
        verts_final = verts_final[:, self.uv_head_idx, :]
        rot_delta = rot_delta[:, self.uv_head_idx, :]
        scale_coef = scale_coef[:, self.uv_head_idx, :]

        return verts_final, rot_delta, scale_coef
    
    
    def capture(self):
        return (
            self.deformNet.state_dict(),
            self.optimizer.state_dict(),
        )
    
    def restore(self, model_args):
        (net_dict,
         opt_dict) = model_args
        self.deformNet.load_state_dict(net_dict)
        self.training_setup()
        self.optimizer.load_state_dict(opt_dict)

    
    def training_setup(self):
        params_group = [
            {'params': self.deformNet.parameters(), 'lr': 1e-4},
        ]
        self.optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999))

    
    def get_template(self):
        geometry_template = self.flame_model.forward_geo(
            self.default_shape_code,
            self.default_expr_code,
            self.default_jaw_pose,
            eye_pose_params=self.default_eyes_pose,
        )

        return geometry_template


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
    
class MLP2(nn.Module):
    def __init__(self, input_dim, condition_dim, output_dim1, output_dim2, hidden_dim=256, hidden_layers=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.hidden_layers = hidden_layers
        mid_layers = math.ceil(hidden_layers/2.)
        self.input_dim = input_dim
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2

        self.fcs1 = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) if i!=(mid_layers-1) else nn.Linear(hidden_dim, hidden_dim+output_dim1) for i in range(mid_layers)]
        )
        self.fcs2 = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) if i!=0 else nn.Linear(hidden_dim+condition_dim, hidden_dim) for i in range(mid_layers, hidden_layers-1)]
        )
        self.output_linear = nn.Linear(hidden_dim, output_dim2)

    def forward(self, input, condition):
        # input: B,V,d1
        # condition: B,d2
        batch_size, N_v, input_dim = input.shape
        input_ori = input.reshape(batch_size*N_v, -1)
        h = input_ori
        for i, l in enumerate(self.fcs1):
            h = self.fcs1[i](h)
            h = F.relu(h)
        oup1 = h[:, -self.output_dim1:]
        h = h[:, :-self.output_dim1]
        ...
        for i, l in enumerate(self.fcs2):
            h = self.fcs1[i](h)
            h = F.relu(h)
        # input_ori = input.reshape(batch_size*N_v, -1)
        # h = input_ori
        # for i, l in enumerate(self.fcs):
        #     h = self.fcs[i](h)
        #     h = F.relu(h)
        # output = self.output_linear(h)
        # output = output.reshape(batch_size, N_v, -1)


class SIRENMLP(nn.Module):
    def __init__(self,
                 input_dim=3,
                 output_dim=3,
                 hidden_dim=256,
                 hidden_layers=8,
                 condition_dim=100,
                 device=None):
        super().__init__()

        self.device = device
        self.input_dim = input_dim
        self.z_dim = condition_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [FiLMLayer(self.input_dim, self.hidden_dim)] +
            [FiLMLayer(self.hidden_dim, self.hidden_dim) for i in range(self.hidden_layers - 1)]
        )
        self.final_layer = nn.Linear(self.hidden_dim, self.output_dim)

        self.mapping_network = MappingNetwork(condition_dim, 256,
                                              len(self.network) * self.hidden_dim * 2)

        self.network.apply(frequency_init(25))
        # self.final_layer.apply(frequency_init(25))
        self.final_layer.weight.data.normal_(0.0, 0.)
        self.final_layer.bias.data.fill_(0.)
        self.network[0].apply(first_layer_film_sine_init)

    def forward_vector(self, input, z):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts):
        frequencies = frequencies * 15 + 30
        x = input

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.final_layer(x)

        return sigma

    def forward(self, vertices, additional_conditioning):
        # vertices: in canonical space of flame
        # vertex eval: torch.Size([N, V, 3])
        # map eval:    torch.Size([N, 3, H, W])
        # conditioning
        # torch.Size([N, C])

        # vertex inputs (N, V, 3) -> (N, 3, V, 1)
        vertices = vertices.permute(0, 2, 1)[:, :, :, None]
        b, c, h, w = vertices.shape

        # to vector
        x = vertices.permute(0, 2, 3, 1).reshape(b, -1, c)

        z = additional_conditioning  # .repeat(1, h*w, 1)

        # apply siren network
        o = self.forward_vector(x, z)

        # to image
        o = o.reshape(b, h, w, self.output_dim)
        output = o.permute(0, 3, 1, 2)

        return output[:, :, :, 0].permute(0, 2, 1)

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift, ignore_conditions=None):
        x = self.layer(x)
        if ignore_conditions is not None:
            cond_freq, cond_phase_shift = freq[:-1], phase_shift[:-1]
            cond_freq = cond_freq.unsqueeze(1).expand_as(x).clone()
            cond_phase_shift = cond_phase_shift.unsqueeze(1).expand_as(x).clone()

            ignore_freq, ignore_phase_shift = freq[-1:], phase_shift[-1:]
            ignore_freq = ignore_freq.unsqueeze(1).expand_as(x)
            ignore_phase_shift = ignore_phase_shift.unsqueeze(1).expand_as(x)

            cond_freq[:, ignore_conditions] = ignore_freq[:, ignore_conditions]
            cond_phase_shift[:, ignore_conditions] = ignore_phase_shift[:, ignore_conditions]
            freq, phase_shift = cond_freq, cond_phase_shift

        else:
            freq = freq.unsqueeze(1).expand_as(x)
            phase_shift = phase_shift.unsqueeze(1).expand_as(x)

        # print('x', x.shape)
        # print('freq', freq.shape)
        # print('phase_shift', phase_shift.shape)
        # x torch.Size([6, 5023, 256])
        # freq torch.Size([6, 5023, 256])
        # phase_shift torch.Size([6, 5023, 256])
        return torch.sin(freq * x + phase_shift)
    
def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
            elif isinstance(m, nn.Conv2d):
                num_input = torch.prod(
                    torch.tensor(m.weight.shape[1:], device=m.weight.device)).cpu().item()
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)

    return init

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2:]

        return frequencies, phase_shifts
    
def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)
        elif isinstance(m, nn.Conv2d):
            num_input = torch.prod(
                torch.tensor(m.weight.shape[1:], device=m.weight.device)).cpu().item()
            m.weight.uniform_(-1 / num_input, 1 / num_input)
