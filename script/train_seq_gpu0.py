import os
import subprocess

id_mica_list = ["id4","id7","id8"]

# mica train
for id_name in id_mica_list:
    command = f'CUDA_VISIBLE_DEVICES="0" /home/xylem/IBC24/CPEM/CPEM-env/bin/python train.py --idname {id_name}'
    
    process = subprocess.run(command, shell=True)
    if process.returncode != 0:
        print(f"Error running {id_name} for mica train")
        continue
    print(f"Success running {id_name} for mica train")

# emoca train use_detail=False
id_emoca_list = ["id4","id5","id7","id8"]

for id_name in id_emoca_list:
    command = f'CUDA_VISIBLE_DEVICES="0" /home/xylem/IBC24/CPEM/CPEM-env/bin/python train_emoca.py --idname {id_name}_emoca --logname log_emoca'
    
    process = subprocess.run(command, shell=True)
    if process.returncode != 0:
        print(f"Error running {id_name} for emoca train")
        continue
    print(f"Success running {id_name} for emoca train")

# # emoca train use_detail=True
# for id_name in id_emoca_list:
#     command = f'CUDA_VISIBLE_DEVICES="1" /home/xylem/IBC24/CPEM/CPEM-env/bin/python train_emoca.py --idname {id_name} --use_detail True --logname log_emoca_detail'
    
#     process = subprocess.run(command, shell=True)
#     if process.returncode != 0:
#         print(f"Error running {id_name} for emoca train")
#         continue
#     print(f"Success running {id_name} for emoca train")

# # smirks train\
# id_smirks_list = ["id4","id5","id7","id8"]
# for id_name in id_smirks_list:
#     command = f'CUDA_VISIBLE_DEVICES="1" /home/xylem/IBC24/CPEM/CPEM-env/bin/python train_smirk.py --idname {id_name} --logname log_smirk'
    
#     process = subprocess.run(command, shell=True)
#     if process.returncode != 0:
#         print(f"Error running {id_name} for smirk train")
#         continue
#     print(f"Success running {id_name} for smirk train")
    
    
    