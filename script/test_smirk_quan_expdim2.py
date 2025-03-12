import os
import subprocess

# region
# id_name_list_mica = ["id4", "id5","Obama","id7", "id8","ljl_wo_glass"]
id_name_list_smirk = ["id2_smirk","id4_smirk", "id5_smirk","id7_smirk", "id8_smirk","Obama_smirk","ljl_wo_glass_smirk"] # obama already tested, smirk_offset
id_name_list_smirk = ["Obama_smirk"]
id_name_list_smirk = ["id5_smirk"]
id_name_list_smirk = ["id2_smirk"]
quan_bit_list = [8,10]
exp_dim_list = [10,20,30,40,50]
# id_name_list_emoca = ["Obama_emoca","id4_emoca", "id5_emoca","id7_emoca", "id8_emoca","ljl_wo_glass_emoca"]
# id_name_list= ["Obama"]


script_name = "test_smirk_quan.py"
for idname in id_name_list_smirk:
    print("idname",idname)
    for quan_bit in quan_bit_list:
        
        for exp_dim in exp_dim_list:
            results = []
           
            logname =f"log_quan_{quan_bit}_exp_dim_{str(exp_dim)}"
            
            base_checkpoint_path = f"/home/xylem/IBC24/FlashAvatar-code/dataset/{idname}/{logname}/ckpt/"
            
            chkpnts = list(range(10000, 150001, 10000))  # 生成原始范围列表
            chkpnts.append(135000)                      # 添加 135000
            chkpnts = sorted(chkpnts)                   # 排序
            # 循环从 25000 到 150000，步长 5000
            for chkpnt in chkpnts:
                
                video1_path = f"/home/xylem/IBC24/FlashAvatar-code/dataset/{idname}/{logname}/rec_{quan_bit}bit_exp_dim{exp_dim}_{chkpnt}_frame1000.avi"
                video2_path = f"/home/xylem/IBC24/FlashAvatar-code/dataset/{idname}/{logname}/ori_frame1000.avi"
                checkpoint_file = os.path.join(base_checkpoint_path, f"chkpnt{chkpnt}.pth")
                print(f"Running checkpoint: {checkpoint_file}")

                # 运行 test_smirk.py
                command = f'CUDA_VISIBLE_DEVICES="2" /home/xylem/IBC24/CPEM/CPEM-env/bin/python {script_name} --idname {idname} --checkpoint {checkpoint_file} --logname {logname} --quan_bit {quan_bit} --exp_dim {exp_dim} --chkpnt_number {chkpnt}'
                print("command",command)
                process = subprocess.run(command, shell=True)
                if process.returncode != 0:
                    print(f"Error running {script_name} for checkpoint {checkpoint_file}")
                    continue

                # 使用 ffmpeg 比较 PSNR
                ffmpeg_command = f'ffmpeg -i {video1_path} -i {video2_path} -filter_complex "[0][1]psnr" -f null -'
                print(ffmpeg_command)
                ffmpeg_process = subprocess.run(ffmpeg_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if ffmpeg_process.returncode != 0:
                    print(f"Error running ffmpeg for checkpoint {checkpoint_file}")
                    continue

                
                # 提取 PSNR (Y通道)
                output = ffmpeg_process.stderr
                psnr_line = next((line for line in output.splitlines() if "PSNR" in line), None)
                if psnr_line:
                    # 提取 Y 通道的 PSNR 值
                    psnr_y = float(psnr_line.split("y:")[1].split(" ")[0])
                    print(f"idname {idname}, quan_bits={quan_bit}, exp_dim={exp_dim}, Checkpoint {chkpnt} :PSNR Y = {psnr_y}")
                    results.append((idname, quan_bit, exp_dim, chkpnt, psnr_y))
                else:
                    print(f"Could not extract PSNR for checkpoint {checkpoint_file}")
            # 找出psnr最大的checkpoint
            max_psnr = max(results, key=lambda x: x[4])
            
            # 输出结果
            print(f"\nFinal Results of idname {idname}, quan_bits={quan_bit}, exp_dim={exp_dim}:")
            for idname,quan_bit, exp_dim, chkpnt, psnr in results:
                print(f"Checkpoint: {chkpnt}, PSNR Y: {psnr}")
            print(f"Max PSNR: {max_psnr[4]} at checkpoint {max_psnr[3]}")

            # 保存到文件
            output_file = os.path.dirname(base_checkpoint_path) + f"/quan_exp_dim_psnr_results_frame1000.txt"
            # output_file = "psnr_results.txt"
            with open(output_file, "w") as f:
                for idname,quan_bit, exp_dim, chkpnt, psnr in results:
                    f.write(f"{idname}, {quan_bit}, {chkpnt}, {psnr}\n")
                # 写入最大psnr以及对应的checkpoint
                f.write(f"Max PSNR: {max_psnr[4]} at checkpoint {max_psnr[3]}")

            print(f"Results saved to {output_file}")

# endregion

