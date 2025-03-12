import torch
id_name_list = ["Obama","id4","id5","id7","id8"]

# for id_name in id_name_list:
    
#     checkpoint = f"/home/xylem/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk/ckpt/chkpnt150000.pth"
#     (model_params, gauss_params, first_iter) = torch.load(checkpoint, weights_only=True)

#     (active_sh_degree, xyz, features_dc, features_rest, scaling_base, rotation_base, opacity,max_radii2D, xyz_gradient_accum, denom,opt_dict, spatial_lr_scale) =gauss_params
#     print(id_name,xyz.shape)
    
#     simplified_gaussian = (active_sh_degree,features_dc, features_rest, scaling_base, rotation_base, opacity)
#     full_gaussian = gauss_params
#     train_gaussian =(max_radii2D, xyz_gradient_accum, denom,opt_dict, spatial_lr_scale)
#     torch.save(train_gaussian, "./pretrained_model/"+ id_name+"_train_param.pth")
#     torch.save(opt_dict,"./pretrained_model/"+ id_name+"_opt_dict.pth")
#     # torch.save(full_gaussian, "./pretrained_model/"+ id_name+"_full_gaussian.pth")
#     # torch.save(simplified_gaussian, "./pretrained_model/"+ id_name+"_simplifed_gaussian.pth")

checkpoint = "/home/xylem/IBC24/FlashAvatar-code/dataset/Obama_smirk/log_smirk/ckpt/chkpnt150000.pth"
(model_params, gauss_params, first_iter) = torch.load(checkpoint)
(deform_net, deform_opt) = model_params
torch.save(deform_net, "./pretrained_model/Obama_deform_net.pth")
torch.save(deform_opt, "./pretrained_model/Obama_deform_opt.pth")
