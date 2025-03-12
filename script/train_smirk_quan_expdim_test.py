import os
import subprocess



# # emoca train use_detail=False
# id_emoca_list = ["id4","id5","id7","id8","Obama"]
exp_dim_list = {10,20,30,40,50}
exp_dim_list = {10}
smirk_id_list = ["id2_smirk","id4_smirk","id5_smirk"] # ,"id5_smirk","id7_smirk","id8_smirk","Obama_smirk","ljl_wo_glass_smirk"]
smirk_id_list = ["Obama_smirk"]
quan_bit_list = [0]
sh_degree = 3
smooth = True


# exp_dim_list = {50}
# # smirk_id_list = ["id2_smirk","id4_smirk"] # ,"id5_smirk","id7_smirk","id8_smirk","Obama_smirk","ljl_wo_glass_smirk"]
# smirk_id_list = ["Obama_smirk"]
# quan_bit_list = [0]

for smirk_id in smirk_id_list:
    chpt_path = f"/home/xylem/IBC24/FlashAvatar-code/dataset/{smirk_id}/log_quan_0_exp_dim_10/ckpt/chkpnt150000.pth"
    for quan_bit in quan_bit_list:
        for exp_dim in exp_dim_list:
            command = f'CUDA_VISIBLE_DEVICES="3" /home/xylem/IBC24/CPEM/CPEM-env/bin/python train_smirk_quan.py --idname {smirk_id} \
            --exp_dim {exp_dim} --logname log_quan_{quan_bit}_exp_dim_{str(exp_dim)}_var --quan_bit {quan_bit} --sh {sh_degree} \
            --smooth {smooth} --start_checkpoint {chpt_path}'
            
            process = subprocess.run(command, shell=True)
            if process.returncode != 0:
                print(f"Error running {smirk_id} for quan_bit_{quan_bit}_exp_dim_{exp_dim} train")
                continue
            print(f"success running {smirk_id} for quan_bit_{quan_bit}_exp_dim_{exp_dim} train")
        