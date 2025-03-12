import os
import subprocess



# # emoca train use_detail=False
# id_emoca_list = ["id4","id5","id7","id8","Obama"]
exp_dim_list = {20,30}
# smirk_id_list = ["id2_smirk","id4_smirk"] # ,"id5_smirk","id7_smirk","id8_smirk","Obama_smirk","ljl_wo_glass_smirk"]
# smirk_id_list = ["id5_smirk","id7_smirk"]
smirk_id_list = ["ljl_wo_glass_smirk","id8_smirk","id7_smirk"]
smirk_id_list = ["Obama_smirk"]
quan_bit_list = [0]


for smirk_id in smirk_id_list:
    for quan_bit in quan_bit_list:
        for exp_dim in exp_dim_list:
            command = f'CUDA_VISIBLE_DEVICES="1" /home/xylem/IBC24/CPEM/CPEM-env/bin/python train_smirk_quan.py --idname {smirk_id} --exp_dim {exp_dim} --logname log_quan_{quan_bit}_exp_dim_{str(exp_dim)} --quan_bit {quan_bit}'
            
            process = subprocess.run(command, shell=True)
            if process.returncode != 0:
                print(f"Error running {smirk_id} for quan_bit_{quan_bit}_exp_dim_{exp_dim} train")
                continue
            print(f"Error running {smirk_id} for quan_bit_{quan_bit}_exp_dim_{exp_dim} train")
        