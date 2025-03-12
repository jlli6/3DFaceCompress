import os
import subprocess

#和 train_test_emoca 一样，这个就不用了
id_emoca_list = ["id4_emoca","id5_emoca","id7_emoca","id8_emoca","Obama_emoca","ljl_wo_glass_emoca"]


for id_name in id_emoca_list:
    command = f'CUDA_VISIBLE_DEVICES="0" /home/xylem/IBC24/CPEM/CPEM-env/bin/python train_emoca.py --idname {id_name} --logname log_emoca --flame mica'
    
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
    
    
    