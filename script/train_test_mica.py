import os
import subprocess

# mica train
id_mica_list = ["id2"]
for id_name in id_mica_list:
    command = f'CUDA_VISIBLE_DEVICES="0" /home/xylem/IBC24/CPEM/CPEM-env/bin/python train.py --idname {id_name}'
    
    process = subprocess.run(command, shell=True)
    if process.returncode != 0:
        print(f"Error running {id_name} for mica train")
        continue
    print(f"Success running {id_name} for mica train")
    
# mica test
for idname in id_mica_list:
    print("testing", idname)
    # 配置路径和参数
    script_name = "test.py"
    # script_name = "test_smirk.py"
    # idname = "id5"
    logname = "log"
    base_checkpoint_path = f"/home/xylem/IBC24/FlashAvatar-code/dataset/{idname}/{logname}/ckpt/"
    video1_path = f"/home/xylem/IBC24/FlashAvatar-code/dataset/{idname}/{logname}/rec.avi"
    video2_path = f"/home/xylem/IBC24/FlashAvatar-code/dataset/{idname}/{logname}/ori.avi"

    # 结果列表
    results = []

    # 循环从 25000 到 150000，步长 5000
    for chkpnt in range(55000, 150001, 5000):
        checkpoint_file = os.path.join(base_checkpoint_path, f"chkpnt{chkpnt}.pth")
        print(f"Running checkpoint: {checkpoint_file}")

        # 运行 test_smirk.py
        command = f'CUDA_VISIBLE_DEVICES="0" /home/xylem/IBC24/CPEM/CPEM-env/bin/python {script_name} --idname {idname} --checkpoint {checkpoint_file} --logname {logname}'
        process = subprocess.run(command, shell=True)
        if process.returncode != 0:
            print(f"Error running {script_name} for checkpoint {checkpoint_file}")
            continue

        # 使用 ffmpeg 比较 PSNR
        ffmpeg_command = f'ffmpeg -i {video1_path} -i {video2_path} -filter_complex "[0][1]psnr" -f null -'
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
            print(f"Checkpoint {chkpnt}: PSNR Y = {psnr_y}")
            results.append((chkpnt, psnr_y))
        else:
            print(f"Could not extract PSNR for checkpoint {checkpoint_file}")

    # 输出结果
    print("\nFinal Results:")
    for chkpnt, psnr in results:
        print(f"Checkpoint: {chkpnt}, PSNR Y: {psnr}")

    # 保存到文件
    output_file = os.path.dirname(base_checkpoint_path) + "/psnr_results.txt"
    # output_file = "psnr_results.txt"
    with open(output_file, "w") as f:
        for chkpnt, psnr in results:
            f.write(f"{chkpnt},{psnr}\n")

    print(f"Results saved to {output_file}")
