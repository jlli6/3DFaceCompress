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
import torch.nn as nn
import sys
from datetime import datetime
import numpy as np
import random

import pickle
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes

eps = 1e-7
class HardConcrete(torch.nn.Module):

    def __init__(self, gamma=1.05, eta=-.05, temperature=2./3.):
        super(HardConcrete, self).__init__()
        self.gamma = gamma
        self.eta = eta
        self.log_alpha = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        self.temperature = torch.nn.Parameter(torch.tensor(temperature), requires_grad=True)

    def forward(self, x):
        # Clamp x to [0,1]
        x = torch.clamp(x, eps, 1-eps)
        # x = (x-torch.mean(x,dim=0)+0.5)/torch.max(1,torch.std(x,dim=0))
        gate = torch.sigmoid((torch.log(x) - torch.log(1-x) + self.log_alpha) / self.temperature)
        gate = gate * (self.gamma - self.eta) + self.eta
        return torch.clamp(gate, 0, 1)
    
    def invert(self,gate):
        # Invert the gate
        with torch.no_grad():
            x = (gate-self.eta)/(self.gamma-self.eta)
            x = inverse_sigmoid(x)
            x = torch.sigmoid(-self.log_alpha+x*self.temperature)
            return x
        
def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    
def PILtoTensor(pil_image):
    resized_image = torch.from_numpy(np.array(pil_image)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))




def quatProduct_batch(q1, q2):
    r1 = q1[:,0] # [B]
    r2 = q2[:,0]
    v1 = torch.stack((q1[:,1], q1[:,2], q1[:,3]), dim=-1) #[B,3]
    v2 = torch.stack((q2[:,1], q2[:,2], q2[:,3]), dim=-1)

    r = r1 * r2 - torch.sum(v1*v2, dim=1) # [B]
    v = r1.unsqueeze(1) * v2 + r2.unsqueeze(1) * v1 + torch.cross(v1, v2) #[B,3]
    q = torch.stack((r, v[:,0], v[:,1], v[:,2]), dim=1)

    return q

def load_binary_pickle(filepath):
    with open(filepath, 'rb') as f:
        if sys.version_info >= (3, 0):
            data = pickle.load(f, encoding='latin1')
        else:
            data = pickle.load(f)
    return data

def a_in_b_torch(a, b):
    ainb = torch.isin(a, b)
    return ainb

def normalize_for_percep(input, mod_n = 64):
    h, w = input.shape[1:3]
    # delta_h = ((h-1)//mod_n + 1)*mod_n - h
    # delta_w = ((w-1)//mod_n + 1)*mod_n - w
    # input_padded = torch.nn.functional.pad(input, (delta_w//2, delta_w-delta_w//2, delta_h//2, delta_h-delta_h//2))
    return input*2.-1.

# borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
def face_vertices_gen(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])

    nd = vertices.shape[2]
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, nd))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]

def dict2obj(d):
    # if isinstance(d, list):
    #     d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    class C(object):
        pass

    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o

class DecayScheduler(object):
    '''A simple class for decaying schedule of various hyperparameters.'''

    def __init__(self, total_steps, decay_name='fix', start=0, end=0, params=None):
        self.decay_name = decay_name
        self.start = start
        self.end = end
        self.total_steps = total_steps
        self.params = params

    def __call__(self, step):
        if self.decay_name == 'fix':
            return self.start
        elif self.decay_name == 'linear':
            if step>self.total_steps:
                return self.end
            return self.start + (self.end - self.start) * step / self.total_steps
        elif self.decay_name == 'exp':
            if step>self.total_steps:
                return self.end
            return max(self.end, self.start*(np.exp(-np.log(1/self.params['temperature'])*step/self.total_steps/self.params['decay_period'])))
            # return self.start * (self.end / self.start) ** (step / self.total_steps)
        elif self.decay_name == 'inv_sqrt':
            return self.start * (self.total_steps / (self.total_steps + step)) ** 0.5
        elif self.decay_name == 'cosine':
            if step>self.total_steps:
                return self.end
            return self.end + 0.5 * (self.start - self.end) * (1 + math.cos(step / self.total_steps * math.pi))
        else:
            raise ValueError('Unknown decay name: {}'.format(self.decay_name))
        
class Pytorch3dRasterizer(nn.Module):
    """  Borrowed from https://github.com/facebookresearch/pytorch3d
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        self.raster_settings_dict = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin': None,
            'perspective_correct': False,
        }
        self.raster_settings = dict2obj(self.raster_settings_dict)

    def forward(self, vertices, faces, attributes=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        raster_settings = self.raster_settings
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )

        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)

        return pixel_vals, pix_to_face, bary_coords #, vismask

    def extra_repr(self):
        return '{image_size}px, blur_radius={blur_radius}, faces_per_pixel={faces_per_pixel}'.format(
            **self.raster_settings_dict)


class Embedder(nn.Module):
    def __init__(self, N_freqs, input_dims=3, include_input=True) -> None:
        super().__init__()
        self.log_sampling = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq = N_freqs - 1
        self.N_freqs = N_freqs
        self.include_input = include_input
        self.input_dims = input_dims
        embed_fns = []
        if self.include_input:
            embed_fns.append(lambda x: x)

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0.,
                                            self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(
                2.**0., 2.**self.max_freq, steps=self.N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
        self.embed_fns = embed_fns
        self.dim_embeded = self.input_dims*len(self.embed_fns)

    def forward(self, inputs, alpha = 10.):
        output = torch.cat([fn(inputs) for fn in self.embed_fns], 2)
        start = 0
        # print(alpha)
        # if self.include_input:
        #     start = 1
        # for i in range(output.shape[1]//2):
        #     output[:, (2*i+start)*self.input_dims:(2*(i+1)+start)*self.input_dims] *= (1-math.cos(math.pi*(max(min(alpha-i, 1.), 0.))))*.5
        return output
    
class CompressedLatents(object):

    def compress(self, latent):
        import torchac
        assert latent.dim() == 2, "Latent should be 2D"
        self.num_latents, self.latent_dim = latent.shape
        flattened = latent.flatten()

        weight = torch.round(flattened).int()
        unique_vals, counts = torch.unique(weight, return_counts = True)
        probs = counts/torch.sum(counts)
        tail_idx = torch.where(probs <= 1.0e-4)[0]
        tail_vals = unique_vals[tail_idx]
        self.tail_locs = {}
        for val in tail_vals:
            self.tail_locs[val.item()] = torch.where(weight == val)[0].detach().cpu()
            weight[weight == val] = unique_vals[counts.argmax()]
        unique_vals, counts = torch.unique(weight, return_counts = True)
        probs = counts/torch.sum(counts)
        weight = weight.detach().cpu()

        cdf = torch.cumsum(probs,dim=0)
        cdf = torch.cat((torch.Tensor([0.0]).to(cdf),cdf))
        cdf = cdf/cdf[-1:] # Normalize the final cdf value just to keep torchac happy
        cdf = cdf.unsqueeze(0).repeat(flattened.size(0),1)
        
        mapping = {val.item():idx.item() for val,idx in zip(unique_vals,torch.arange(unique_vals.shape[0]))}
        self.mapping = mapping
        weight.apply_(mapping.get)
        byte_stream = torchac.encode_float_cdf(cdf.detach().cpu(), weight.to(torch.int16))
        
        self.byte_stream, self.mapping, self.cdf = byte_stream, mapping, cdf[0].detach().cpu().numpy()

    def uncompress(self):
        import torchac
        cdf = torch.tensor(self.cdf).unsqueeze(0).repeat(self.num_latents*self.latent_dim,1)
        weight = torchac.decode_float_cdf(cdf, self.byte_stream)
        weight = weight.to(torch.float32)
        # weight = self.tail_decode(cdf, self.byte_stream, self.tail_vals, self.tail_idx)
        # weight = weight.to(torch.float32)
        inverse_mapping = {v:k for k,v in self.mapping.items()}
        weight.apply_(inverse_mapping.get)
        for val, locs in self.tail_locs.items():
            weight[locs] = val
        weight = weight.view(self.num_latents, self.latent_dim)
        return weight
    
    
class CompressedLatents_quantize(object):

    def compress(self, latent):
        import torchac
        assert latent.dim() == 2, "Latent should be 2D"
        self.num_latents, self.latent_dim = latent.shape
        flattened = latent.flatten()

        # 16-bit quantization
        min_val, max_val = flattened.min(), flattened.max()
        self.min_val, self.max_val = min_val.item(), max_val.item()
        quantization_levels = 2 ** 10 - 1
        weight = ((flattened - min_val) / (max_val - min_val) * quantization_levels).round().int()

        unique_vals, counts = torch.unique(weight, return_counts=True)
        probs = counts / torch.sum(counts)
        tail_idx = torch.where(probs <= 1.0e-10)[0]
        tail_vals = unique_vals[tail_idx]
        self.tail_locs = {}
        for val in tail_vals:
            self.tail_locs[val.item()] = torch.where(weight == val)[0].detach().cpu()
            weight[weight == val] = unique_vals[counts.argmax()]
        unique_vals, counts = torch.unique(weight, return_counts=True)
        probs = counts / torch.sum(counts)
        weight = weight.detach().cpu()

        cdf = torch.cumsum(probs, dim=0)
        cdf = torch.cat((torch.Tensor([0.0]).to(cdf), cdf))
        cdf = cdf / cdf[-1:]  # Normalize the final cdf value just to keep torchac happy
        cdf = cdf.unsqueeze(0).repeat(flattened.size(0), 1)

        mapping = {val.item(): idx.item() for val, idx in zip(unique_vals, torch.arange(unique_vals.shape[0]))}
        self.mapping = mapping
        weight.apply_(mapping.get)
        byte_stream = torchac.encode_float_cdf(cdf.detach().cpu(), weight.to(torch.int16))

        self.byte_stream, self.mapping, self.cdf = byte_stream, mapping, cdf[0].detach().cpu().numpy()
        
        # # 将byte_stream, mapping, cdf分别保存为pkl
        # with open(f"byte_stream_{self.num_latents}_{self.latent_dim}.pkl", 'wb') as f:
        #     pickle.dump(self.byte_stream, f)
        # with open(f"mapping_{self.num_latents}_{self.latent_dim}.pkl", 'wb') as f:
        #     pickle.dump(self.mapping, f)
        # with open(f"cdf_{self.num_latents}_{self.latent_dim}.pkl", 'wb') as f:
        #     pickle.dump(self.cdf, f)

    def uncompress(self):
        import torchac
        cdf = torch.tensor(self.cdf).unsqueeze(0).repeat(self.num_latents * self.latent_dim, 1)
        weight = torchac.decode_float_cdf(cdf, self.byte_stream)
        weight = weight.to(torch.float32)
        
        inverse_mapping = {v: k for k, v in self.mapping.items()}
        weight.apply_(inverse_mapping.get)
        for val, locs in self.tail_locs.items():
            weight[locs] = val

        # Dequantization: Map back to the original floating-point range
        quantization_levels = 2 ** 10 - 1
        weight = weight / quantization_levels * (self.max_val - self.min_val) + self.min_val

        weight = weight.view(self.num_latents, self.latent_dim)
        return weight
