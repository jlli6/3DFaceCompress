import os
import subprocess



# # emoca train use_detail=False
# id_emoca_list = ["id4","id5","id7","id8","Obama"]
exp_dim_list = {10,20,30,40}
id_name = "Obama"
flame_type = "mica"

for exp_dim in exp_dim_list:
    command = f'CUDA_VISIBLE_DEVICES="2" /home/xylem/IBC24/CPEM/CPEM-env/bin/python train_emoca.py --idname {id_name}_emoca --exp_dim {exp_dim} --logname log_{flame_type}_exp_dim_new_{str(exp_dim)} --flame {flame_type}'
    
    process = subprocess.run(command, shell=True)
    if process.returncode != 0:
        print(f"Error running {id_name} for emoca train")
        continue
    print(f"Success running {id_name} for emoca train")

# # emoca train use_detail=False
# id_emoca_list = ["id4","id5","id7","id8"]

# for id_name in id_emoca_list:
#     command = f'CUDA_VISIBLE_DEVICES="0" /home/xylem/IBC24/CPEM/CPEM-env/bin/python train_emoca.py --idname {id_name}_emoca --logname log_emoca'
    
#     process = subprocess.run(command, shell=True)
#     if process.returncode != 0:
#         print(f"Error running {id_name} for emoca train")
#         continue
#     print(f"Success running {id_name} for emoca train")
    
# # smirks train\
# id_smirks_list = ["id4","id5","id7","id8"]
# for id_name in id_smirks_list:
#     command = f'CUDA_VISIBLE_DEVICES="1" /home/xylem/IBC24/CPEM/CPEM-env/bin/python train_smirk.py --idname {id_name}_smirk --logname log_smirk'
    
#     process = subprocess.run(command, shell=True)
#     if process.returncode != 0:
#         print(f"Error running {id_name} for smirk train")
#         continue
#     print(f"Success running {id_name} for smirk train")
    
    
    