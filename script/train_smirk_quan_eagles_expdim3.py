import os
import subprocess



# # emoca train use_detail=False
# id_emoca_list = ["id4","id5","id7","id8","Obama"]
exp_dim_list = {10,20,30,40,50}
# exp_dim_list = {10}
smirk_id_list = ["id1_smirk","id2_smirk","id4_smirk","id5_smirk","id8_smirk","Obama_smirk","ljl_smirk","xj_smirk"] # ,"id5_smirk","id7_smirk","id8_smirk","Obama_smirk","ljl_wo_glass_smirk"]
smirk_id_list = ["id5_smirk","xj_smirk"]
quan_bit_list = [8,10]
smirk_id_list = ["id5_smirk"]

# exp_dim_list = {50}
# # smirk_id_list = ["id2_smirk","id4_smirk"] # ,"id5_smirk","id7_smirk","id8_smirk","Obama_smirk","ljl_wo_glass_smirk"]
# smirk_id_list = ["Obama_smirk"]
# quan_bit_list = [0]

for smirk_id in smirk_id_list:
    for quan_bit in quan_bit_list:
        for exp_dim in exp_dim_list:
            command = f'CUDA_VISIBLE_DEVICES="3" /home/ljl/workspace/ICME2025/LIC_TCM/tcm_venv/bin/python train_smirk_eagles.py \
                --config configs/efficient-3dgs.yaml \
                --idname {smirk_id} --exp_dim {exp_dim} \
                --logname log_smirk_eagles_mlp6_rest141_purn0_quan{quan_bit}_exp_dim_{str(exp_dim)} \
                --quan_bit {quan_bit} --loss_type lpips --skip_test --iteration 150000 --mlp_layer 6'
            
            process = subprocess.run(command, shell=True)
            if process.returncode != 0:
                print(f"Error running {smirk_id} for quan_bit_{quan_bit}_exp_dim_{exp_dim} train")
                continue
            print(f"successful running {smirk_id} for quan_bit_{quan_bit}_exp_dim_{exp_dim} train")
        