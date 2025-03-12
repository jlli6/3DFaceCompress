
import os
import cv2
import numpy as np
import subprocess

def get_265gt():
    id_list = ["Obama","id4","id5","id7","id8"] #
    id_list = ["id2"]
    test_img_num =1000
    img_start_num = 1
    for id_name in id_list:
        id_image_folder = f"/home/xylem/IBC24/FlashAvatar-code/dataset/{id_name}/imgs"
        id_mask_folder = f"/home/xylem/IBC24/FlashAvatar-code/dataset/{id_name}/parsing"
        save_folder = f"/home/xylem/IBC24/FlashAvatar-code/experimet/{id_name}/imgs_parsing"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        for img_num in range(img_start_num, img_start_num+test_img_num):
            
            if id_name == "Obama":
                img_path = os.path.join(id_image_folder, f"{str(img_num).zfill(5)}.jpg")
            else:
                img_path = os.path.join(id_image_folder, f"{str(img_num).zfill(5)}.png")
            img_parsing_path = os.path.join(id_mask_folder, f"{str(img_num).zfill(5)}_neckhead.png")
            
            # img_parsing_path 是人像分割的黑白掩码，白色表示前景，黑色表示背景，根据这个掩码提取img_path 的前景，并且将背景变成白色
            # 加载图像
            img = cv2.imread(img_path)  # 原始彩色图像
            img_parsing = cv2.imread(img_parsing_path)  # 掩码，灰度图

            # 解析掩码，找到白色部分 (R=255, G=255, B=255)
            mask = (img_parsing[:, :, 0] == 255) & (img_parsing[:, :, 1] == 255) & (img_parsing[:, :, 2] == 255)

            # 创建一个纯白图像作为背景
            result_img = np.full_like(img, 255, dtype=np.uint8)

            # 将掩码对应的部分替换为 head_img 中的内容
            result_img[mask] = img[mask]

            # 保存结果
            save_path = os.path.join(save_folder,  f"{str(img_num).zfill(5)}.png")
            cv2.imwrite(save_path, result_img)
        
            print(f"Processed {id_name} {img_num}")

        
import os
import time
import os
import subprocess
from skimage.metrics import structural_similarity as ssim
import lpips
import cv2
import torch 

from PIL import Image
import torchvision.transforms as transforms

lpips_model = lpips.LPIPS(net='alex')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 指定GPU
lpips_model.to(device)

def load_image(image_path):
    # Load an image file into a PyTorch tensor
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to the input size expected by the network
        transforms.ToTensor()           # Transform it into a torch tensor
    ])
    return transform(image).unsqueeze(0).to(device)  # Add a batch dimension

# 步骤4：计算SSIM
def calculate_ssim(imageA_path, imageB_path):
    # Load the images in grayscale
    imageA = cv2.imread(imageA_path, cv2.IMREAD_GRAYSCALE)
    imageB = cv2.imread(imageB_path, cv2.IMREAD_GRAYSCALE)

    # Compute the SSIM between the two images
    score, diff = ssim(imageA, imageB, full=True)
    # print("SSIM: {}".format(score))
    return score
    

    
# 步骤5：计算LPIPS
def calculate_lpips(imageA_path, imageB_path):
    # Initialize the LPIPS model; you can choose from 'alex', 'vgg', etc.
   

    # Load images
    imageA = load_image(imageA_path)
    imageB = load_image(imageB_path)

    # Use the model to compute LPIPS distance
    distance = lpips_model(imageA, imageB)

    # print("LPIPS Distance: {:.4f}".format(distance.item()))
    return distance.item()

# 步骤6：计算VMAF
def calculate_vmaf(ref_video, dis_video):
    command = f"vmaf {ref_video} {dis_video} --model vmaf_v0.6.1.pkl"
    output = subprocess.check_output(command, shell=True)
    output = output.decode("utf-8")
    vmaf_score = float(output.split("\n")[-2].split(",")[-1].split(":")[-1])
    return vmaf_score


# 将rgb图片序列转换为yuv
def rgb_yuv():
    
    id_list = ["Obama","id4","id5","id7","id8"] #
    id_list = ["id2"]
    # 保存当前工作目录
    original_path = os.getcwd()

    for id in id_list:
        
        id_path_gt = f"/home/xylem/IBC24/FlashAvatar-code/experimet/{id}/imgs_parsing"

        # 改变当前工作目录到地面实况路径
        os.chdir(id_path_gt)
        cmd_gt = "ffmpeg -y -f image2 -framerate 25 -s 512x512 -i %05d.png -c:v rawvideo -pix_fmt yuv420p ../gt_25fps.yuv"
        os.system(cmd_gt)

        # 改回原来的工作目录
        os.chdir(original_path)
        
 
def write_psnr_to_file(id, bitrate, psnr_value):
    with open(f"/home/xylem/IBC24/FlashAvatar-code/experimet/id_bitrate_psnr_{id}.txt", "a") as f:
        f.write(f"PSNR for {id} at {bitrate}k: {psnr_value} \n")   
         
def write_ssim_lpips_to_file(id, bitrate, ssim, lpips):
    with open(f"/home/xylem/IBC24/FlashAvatar-code/experimet/id_bitrate_ssim_lpips_{id}.txt", "a") as f:
        f.write(f"SSIM for {id} at {bitrate}k: {ssim} , LPIPS for {id} at {bitrate}k: {lpips} \n")
        
def write_gs_ssim_lpips_to_file(id, quan_bit,exp_dim, ssim, lpips):
    log_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/experiment/gs_ssmi_lpips/ssim_lpips_compressed_best_frame500.txt"
    # log_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/experiment/gs_ssmi_lpips/ssim_lpips_Obama_purn0.txt"
    # log_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/experiment/gs_ssmi_lpips/ssim_lpips_manual.txt"
    # log_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/experiment/gs_ssmi_lpips/ssim_lpips_purn40_uncompressed_frame500.txt"
    # log_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/experiment/gs_ssmi_lpips/ssim_lpips_purn0_uncompressed_frame500.txt"
    with open(log_path, "a") as f:
        f.write(f"{id} at {quan_bit}bit_{exp_dim}dim, SSIM: {ssim} , LPIPS: {lpips} \n")
        
# 计算h265编码后的PSNR
def yuv_265():
    id_list = ["Obama","id4","id5","id7","id8"] #
    # id_list = ["id2"]
    bitrate_list = [5,20,30,40,50,60,70,80,90,100,120,130,150,200]
    bitrate_list = [10,15]
    for id in id_list:

        for bitrate in bitrate_list:
            bitrate_dir = "/home/xylem/IBC24/FlashAvatar-code/experimet/{}/yuv_rgb_{}k".format(id, bitrate)
            os.makedirs(bitrate_dir, exist_ok=True)
            gt_path = f"/home/xylem/IBC24/FlashAvatar-code/experimet/{id}/gt_25fps.yuv"
            os.chdir(bitrate_dir)
            cmd = "ffmpeg -y -s 512x512 -pix_fmt yuv420p -r 25 -i {} -c:v libx265 -x265-params bframes=0 -b:v {}k -r 25 h265_{}k.hevc".format(gt_path, bitrate, bitrate)
            os.system(cmd)
            # print("==================================={}={}================================================".format(id, bitrate))
            # 比较编码后的yuv和原yuv的psnr
            cmd_psnr = "ffmpeg -y -s 512x512 -pix_fmt yuv420p -r 25 -i {} -i h265_{}k.hevc -filter_complex \"[0:v][1:v]psnr\" -f null -".format(gt_path, bitrate)
            # 提取出psnr
             # 使用 subprocess 捕获 PSNR 输出
            process = subprocess.Popen(cmd_psnr, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            # 解析 stderr 输出中的 PSNR 值
            stderr_output = stderr.decode("utf-8")
            psnr_line = [line for line in stderr_output.split("\n") if "PSNR" in line]
            if psnr_line:
                psnr_value = psnr_line[-1].split("average:")[1].split()[0]
                print(f"PSNR for {id} at {bitrate}k: {psnr_value}")
                write_psnr_to_file(id, bitrate, psnr_value)
            else:
                print(f"Failed to extract PSNR for {id} at {bitrate}k")
            
            # os.system(cmd)
            print("*"*50)

# 将编码后的hevc文件转换为rgb图片序列，并且计算SSIM和LPIPS
def hevc_ssim_lpips():
    
    id_list = ["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8"]
    bitrate_list = [5,20,30,40,50]
    
    id_list = ["Obama","id4","id5","id7","id8"]
    bitrate_list = [5,20,30,40,50,60,70,80,90,100]
    
    # bitrate_list =[5]
    id_list = ["id2"]
    
    # h265转成yuv,再转成图片序列
    for id in id_list:
        for bitrate in bitrate_list:
            hevc_path = "/home/xylem/IBC24/FlashAvatar-code/experimet/{}/yuv_rgb_{}k/h265_{}k.hevc".format(id, bitrate, bitrate)
            yuv_path = "/home/xylem/IBC24/FlashAvatar-code/experimet/{}/yuv_rgb_{}k/h265_{}k.yuv".format(id, bitrate, bitrate)
            os.chdir("/home/xylem/IBC24/FlashAvatar-code/experimet/{}/yuv_rgb_{}k".format(id, bitrate))
            # 265转yuv命令是ffmpeg -i h265_20k.hevc -c:v rawvideo -pix_fmt yuv420p h265_20k.yuv
            cmd = "ffmpeg -y -i {} -c:v rawvideo -pix_fmt yuv420p {}".format(hevc_path, yuv_path) 
            os.system(cmd)
            
            # yuv生成rgb图片序列命令是:
            #      ffmpeg -f rawvideo -pixel_format yuv420p -video_size 512x512 -i input.yuv -vf "scale=512:512,format=rgb24" -start_number 0000 -frames:v 1000 output_%04d.png
            cmd = "ffmpeg -y -f rawvideo -pixel_format yuv420p -video_size 512x512 -i {} -vf \"scale=512:512,format=rgb24\" -start_number 1 -frames:v 1000 output_%05d.png".format(yuv_path)
            os.system(cmd)
            
    print("**"*50)     
    for bitrate in bitrate_list:
        id_ssim = []
        id_lpips = []
        for id in id_list:
            
        
            # 计算SSIM和LPIPS
            ssim_scores = []
            lpips_scores = []
            for i in range(1,1001):
                gt_image = os.path.join("/home/xylem/IBC24/FlashAvatar-code/experimet/{}/imgs_parsing".format(id), f"{i:05}.png")
                pred_image = os.path.join("/home/xylem/IBC24/FlashAvatar-code/experimet/{}/yuv_rgb_{}k".format(id, bitrate), f"output_{i:05d}.png")
                ssim_score = calculate_ssim(gt_image, pred_image)
                ssim_scores.append(ssim_score)
                lpips_score = calculate_lpips(gt_image, pred_image)
                lpips_scores.append(lpips_score)
            avg_ssim = round(sum(ssim_scores)/len(ssim_scores),4)
            avg_lpips = round(sum(lpips_scores)/len(lpips_scores),4)
            id_ssim.append(avg_ssim)
            id_lpips.append(avg_lpips)
            write_ssim_lpips_to_file(id, bitrate, avg_ssim, avg_lpips)
            print("==================================={}={}================================================".format(id, bitrate))
            print(f"{id}, hevc bitrate {bitrate} SSIM scores:", avg_ssim)
            print(f"{id}, hevc bitrate {bitrate} LPIPS scores:", avg_lpips)
        # print("avg_ssim:", sum(id_ssim)/len(id_ssim))
        # print("avg_lpips:", sum(id_lpips)/len(id_lpips))
  
def id_ssim_lpips():
    # id_list = ["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8"]
   
    id_list = ["id5"]
    id_list = ["id2","id4","id8","id5"]
    frame_num = 500
    # id_list = ["Obama"]
    quan_bit_list = [8,10]
    exp_dim_list = [10,20,30,40,50]
    for quan_bit in quan_bit_list:
        for exp_dim in exp_dim_list:
            for id_name in id_list:
                print(f"==============================start_{id_name}_smirk_{quan_bit}bit_{exp_dim}dim====================================")
                
                # psnr_fram1000_txt = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}/ckpt/quan_exp_dim_psnr_results.txt"
                psnr_fram1000_txt = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}/ckpt/quan_exp_dim_psnr_results_frame500.txt"
                
                # 读取psnr_fram1000_txt最后一行，格式为Max PSNR: 34.281218 at checkpoint 135000，找到最佳的ckpt
                try:
                    with open(psnr_fram1000_txt, "r") as f:
                        lines = f.readlines()
                        print("split :",lines[-1].split(" ") )
                        best_chkpnt = lines[-1].split(" ")[-2]
                        print("find best checkpoint:", best_chkpnt)
                except:
                    print(f"Error reading {psnr_fram1000_txt}")
                    
                # 计算SSIM和LPIPS
                # best_rec_video = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}", f"rec_{quan_bit}bit_exp_dim{exp_dim}_{best_chkpnt}_frame1000.avi")
                # ori_video_path = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}", f"ori_frame1000.avi")
                
                best_rec_video = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}", f"rec_{quan_bit}bit_exp_dim{exp_dim}_{best_chkpnt}_frame500.avi")
                ori_video_path = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}", f"ori_frame500.avi")
                
                
                print("best_rec_video:", best_rec_video)
                print("ori_video_path:", ori_video_path)
                
                # 使用ffmpeg将两个avi视频转成图片序列
                # best_rec_imgs_path = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}", "rec_best_imgs")
                # ori_imgs_path = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}", "ori_imgs")
                
                
                best_rec_imgs_path = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}", "rec_best_imgs_500")
                ori_imgs_path = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}", "ori_imgs_500")
                
                os.makedirs(best_rec_imgs_path, exist_ok=True)
                os.makedirs(ori_imgs_path, exist_ok=True)
                os.system(f"ffmpeg -i {best_rec_video} -vf \"fps=30\" -q:v 2 -start_number 1 {best_rec_imgs_path}/%05d.png")
                os.system(f"ffmpeg -i {ori_video_path} -vf \"fps=30\" -q:v 2 -start_number 1 {ori_imgs_path}/%05d.png")
                
                # ffmpeg -i rec_10bit_exp_dim30_150000_frame1000.avi -vf "fps=25" rec_%05d.png
                
                # 计算SSIM和LPIPS
                ssim_scores = []
                lpips_scores = []
                for i in range(1,1+frame_num):
                    pred_image = os.path.join(best_rec_imgs_path, f"{i:05}.png")
                    gt_image = os.path.join(ori_imgs_path, f"{i:05d}.png")
                    ssim_score = calculate_ssim(gt_image, pred_image)
                    ssim_scores.append(ssim_score)
                    lpips_score = calculate_lpips(gt_image, pred_image)
                    lpips_scores.append(lpips_score)
                avg_ssim = round(sum(ssim_scores)/len(ssim_scores),4)
                avg_lpips = round(sum(lpips_scores)/len(lpips_scores),4)
                print(f"==============================result_of_{id_name}_smirk_{quan_bit}bit_{exp_dim}dim================================================")
                
                print(f"{id_name}_smirk, quan_bit {quan_bit}, exp_dim {exp_dim} SSIM scores:", avg_ssim)
                print(f"{id_name}_smirk, quan_bit {quan_bit}, exp_dim {exp_dim} LPIPS scores:", avg_lpips)
                write_gs_ssim_lpips_to_file(id_name, quan_bit, exp_dim, avg_ssim, avg_lpips)
                
def id_ssim_lpips_purn0():
    
    # id_list = ["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8"]
   
    
    id_list = ["id2","id4","id8","id5","Obama"]
    frame_num = 500
    # id_list = ["Obama"]
    quan_bit_list = [8,10]
    exp_dim_list = [10,20,30,40,50]
    
    id_list = ["Obama"]
    quan_bit_list = [10]
    exp_dim_list = [10]
    for quan_bit in quan_bit_list:
        for exp_dim in exp_dim_list:
            for id_name in id_list:
                print(f"==============================start_{id_name}_smirk_{quan_bit}bit_{exp_dim}dim====================================")
                
                log_name = f"log_smirk_eagles_mlp6_rest141_purn0_quan{quan_bit}_exp_dim_{str(exp_dim)}_retain"
                # psnr_fram1000_txt = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn0_quan{quan_bit}_exp_dim_{str(exp_dim)}/ckpt/quan_exp_dim_psnr_results.txt"
                psnr_fram1000_txt = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/{log_name}/ckpt/quan_exp_dim_psnr_results_frame500.txt"
                
                # 读取psnr_fram1000_txt最后一行，格式为Max PSNR: 34.281218 at checkpoint 135000，找到最佳的ckpt
                try:
                    with open(psnr_fram1000_txt, "r") as f:
                        lines = f.readlines()
                        print("split :",lines[-1].split(" ") )
                        best_chkpnt = lines[-1].split(" ")[-2]
                        print("find best checkpoint:", best_chkpnt)
                except:
                    print(f"Error reading {psnr_fram1000_txt}")
                    
                # 计算SSIM和LPIPS
                # best_rec_video = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn0_quan{quan_bit}_exp_dim_{str(exp_dim)}", f"rec_{quan_bit}bit_exp_dim{exp_dim}_{best_chkpnt}_frame1000.avi")
                # ori_video_path = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn0_quan{quan_bit}_exp_dim_{str(exp_dim)}", f"ori_frame1000.avi")
                
                best_rec_video = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/{log_name}", f"rec_{quan_bit}bit_exp_dim{exp_dim}_{best_chkpnt}_frame500.avi")
                ori_video_path = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/{log_name}", f"ori_frame500.avi")
                
                
                print("best_rec_video:", best_rec_video)
                print("ori_video_path:", ori_video_path)
                
                # 使用ffmpeg将两个avi视频转成图片序列
                # best_rec_imgs_path = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn0_quan{quan_bit}_exp_dim_{str(exp_dim)}", "rec_best_imgs")
                # ori_imgs_path = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn0_quan{quan_bit}_exp_dim_{str(exp_dim)}", "ori_imgs")
                
                
                best_rec_imgs_path = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/{log_name}", "rec_best_imgs_500")
                ori_imgs_path = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/{log_name}", "ori_imgs_500")
                
                os.makedirs(best_rec_imgs_path, exist_ok=True)
                os.makedirs(ori_imgs_path, exist_ok=True)
                os.system(f"ffmpeg -i {best_rec_video} -vf \"fps=30\" -q:v 2 -start_number 1 {best_rec_imgs_path}/%05d.png")
                os.system(f"ffmpeg -i {ori_video_path} -vf \"fps=30\" -q:v 2 -start_number 1 {ori_imgs_path}/%05d.png")
                
                # ffmpeg -i rec_10bit_exp_dim30_150000_frame1000.avi -vf "fps=25" rec_%05d.png
                
                # 计算SSIM和LPIPS
                ssim_scores = []
                lpips_scores = []
                for i in range(1,1+frame_num):
                    pred_image = os.path.join(best_rec_imgs_path, f"{i:05}.png")
                    gt_image = os.path.join(ori_imgs_path, f"{i:05d}.png")
                    ssim_score = calculate_ssim(gt_image, pred_image)
                    ssim_scores.append(ssim_score)
                    lpips_score = calculate_lpips(gt_image, pred_image)
                    lpips_scores.append(lpips_score)
                avg_ssim = round(sum(ssim_scores)/len(ssim_scores),4)
                avg_lpips = round(sum(lpips_scores)/len(lpips_scores),4)
                print(f"==============================result_of_{id_name}_smirk_{quan_bit}bit_{exp_dim}dim================================================")
                
                print(f"{id_name}_smirk, quan_bit {quan_bit}, exp_dim {exp_dim} SSIM scores:", avg_ssim)
                print(f"{id_name}_smirk, quan_bit {quan_bit}, exp_dim {exp_dim} LPIPS scores:", avg_lpips)
                # write_gs_ssim_lpips_to_file(id_name, quan_bit, exp_dim, avg_ssim, avg_lpips)
        
    
       
def delete_yuv():
    
    id_list = ["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8"]
    bitrate_list = [5,20,30,40,50]
    
    id_list = ["Obama","id4","id5","id7","id8"]
    bitrate_list = [20,30,40,50,60,70,80,90,100]
    
    # h265转成yuv,再转成图片序列
    for id in id_list:
        for bitrate in bitrate_list:
            # hevc_path = "/home/xylem/IBC24/FlashAvatar-code/experimet/{}/yuv_rgb_{}k/h265_{}k.hevc".format(id, bitrate, bitrate)
            yuv_path = "/home/xylem/IBC24/FlashAvatar-code/experimet/{}/yuv_rgb_{}k/h265_{}k.yuv".format(id, bitrate, bitrate)
            cmd = f"rm {yuv_path}" 
            os.system(cmd)             

def test_smirk_eagles_compressed_best():
    id_list = ["id2","id4","id8","Obama","id5"]
    quan_bit_list = [8,10]
    exp_dim_list = [10,20,30,40,50]

    # id_list = ["id2"]
    # quan_bit_list = [10]
    # exp_dim_list = [10]
    frame_num = 500
    script_name = "train_smirk_eagles.py"

    results = [] # 保存idname, quan_bit, exp_dim, chkpnt, psnr

    for quan_bit in quan_bit_list:
        for exp_dim in exp_dim_list:
            for id_name in id_list:
                print(f"==============================start_{id_name}_smirk_{quan_bit}bit_{exp_dim}dim====================================")
                
                
                psnr_fram1000_txt = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}/ckpt/quan_exp_dim_psnr_results.txt"
                # 读取psnr_fram1000_txt最后一行，格式为Max PSNR: 34.281218 at checkpoint 135000，找到最佳的ckpt
                logname =f"log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}"
                
                
                try:
                    with open(psnr_fram1000_txt, "r") as f:
                        lines = f.readlines()
                        print("split :",lines[-1].split(" ") )
                        best_chkpnt = lines[-1].split(" ")[-2]
                        print("find best checkpoint:", best_chkpnt)
                except:
                    print(f"Error reading {psnr_fram1000_txt}")
                video1_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/{logname}/rec_{quan_bit}bit_exp_dim{exp_dim}_{best_chkpnt}_frame500_best_compressed.avi"
                video2_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/{logname}/ori_frame500.avi"
                
                
                # 执行过了，跳过
                # #  获得测试集上的结果          
                # command = f'CUDA_VISIBLE_DEVICES="2" /home/ljl/workspace/ICME2025/LIC_TCM/tcm_venv/bin/python train_smirk_eagles.py \
                # --config configs/efficient-3dgs.yaml \
                # --idname {id_name}_smirk --exp_dim {exp_dim} \
                # --logname log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)} \
                # --quan_bit {quan_bit} --loss_type lpips --skip_train --iteration {best_chkpnt} --mlp_layer 6 \
                # --chkpnt_number {best_chkpnt}'
                            
                # print("command",command)
                
                # process = subprocess.run(command, shell=True)
                
                # if process.returncode != 0:
                #     print(f"Error running {script_name} for checkpoint {id_name}, {quan_bit}, {exp_dim}")
                #     continue
                # else:
                #     print(f"Success running {script_name} for checkpoint {id_name}, {quan_bit}, {exp_dim}")

                # # 使用 ffmpeg 比较 PSNR
                # ffmpeg_command = f'ffmpeg -i {video1_path} -i {video2_path} -filter_complex "[0][1]psnr" -f null -'
                # print(ffmpeg_command)
                # ffmpeg_process = subprocess.run(ffmpeg_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                # if ffmpeg_process.returncode != 0:
                #     print(f"Error running ffmpeg for checkpoint {id_name}, {quan_bit}, {exp_dim}")
                #     continue

                
                # # 提取 PSNR (Y通道)
                # output = ffmpeg_process.stderr
                # psnr_line = next((line for line in output.splitlines() if "PSNR" in line), None)
                # if psnr_line:
                #     # 提取 Y 通道的 PSNR 值
                #     psnr_y = float(psnr_line.split("y:")[1].split(" ")[0])
                #     print(f"idname {id_name}, quan_bits={quan_bit}, exp_dim={exp_dim}, Checkpoint {best_chkpnt} :PSNR Y = {psnr_y}")
                #     results.append((id_name, quan_bit, exp_dim, best_chkpnt, psnr_y))
                # else:
                #     print(f"Could not extract PSNR for checkpoint {id_name}, {quan_bit}, {exp_dim}")
                    
                # 比较lpips和ssim
                
                
                # 使用ffmpeg将两个avi视频转成图片序列
                best_rec_imgs_path = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/{logname}", "rec_best_imgs_compressed_500")
                ori_imgs_path = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/{logname}", "ori_imgs_500")
                
                os.makedirs(best_rec_imgs_path, exist_ok=True)
                os.makedirs(ori_imgs_path, exist_ok=True)
            
                os.system(f"ffmpeg -i {video1_path} -vf \"fps=30\" -q:v 2 -start_number 1 {best_rec_imgs_path}/%05d.png")
                
                
                # ffmpeg -i rec_10bit_exp_dim30_150000_frame1000.avi -vf "fps=25" rec_%05d.png
                
                # 计算SSIM和LPIPS
                ssim_scores = []
                lpips_scores = []
                for i in range(1,1+frame_num):
                    pred_image = os.path.join(best_rec_imgs_path, f"{i:05}.png")
                    gt_image = os.path.join(ori_imgs_path, f"{i:05d}.png")
                    ssim_score = calculate_ssim(gt_image, pred_image)
                    ssim_scores.append(ssim_score)
                    lpips_score = calculate_lpips(gt_image, pred_image)
                    lpips_scores.append(lpips_score)
                avg_ssim = round(sum(ssim_scores)/len(ssim_scores),4)
                avg_lpips = round(sum(lpips_scores)/len(lpips_scores),4)
                print(f"==============================result_of_{id_name}_smirk_{quan_bit}bit_{exp_dim}dim================================================")
                
                print(f"{id_name}_smirk, quan_bit {quan_bit}, exp_dim {exp_dim} SSIM scores:", avg_ssim)
                print(f"{id_name}_smirk, quan_bit {quan_bit}, exp_dim {exp_dim} LPIPS scores:", avg_lpips)
                write_gs_ssim_lpips_to_file(id_name, quan_bit, exp_dim, avg_ssim, avg_lpips)
            

    # 保存到文件
    output_file = "/home/ljl/workspace/IBC24/FlashAvatar-code/experiment/compressed_best_results_500.txt"
    # output_file = "psnr_results.txt"
    with open(output_file, "a") as f:
        for idname,quan_bit, exp_dim, chkpnt, psnr in results:
            f.write(f"idname: {idname}, quan_bit:{quan_bit}, exp_dim:{exp_dim}, chkpnt: {chkpnt}, psnr:{round(psnr,2)} dB\n")

    print(f"Results saved to {output_file}")
    
def get_compressed_model():
    id_list = ["id2","id4","id8","Obama","id5"]
    quan_bit_list = [8,10]
    exp_dim_list = [10,20,30,40,50]
    
    # id_list = ["id2"]
    # quan_bit_list = [10]
    # exp_dim_list = [10]

    script_name = "train_smirk_eagles.py"

    results = [] # 保存idname, quan_bit, exp_dim, chkpnt, psnr

    for quan_bit in quan_bit_list:
        for exp_dim in exp_dim_list:
            for id_name in id_list:
                print(f"==============================start_{id_name}_smirk_{quan_bit}bit_{exp_dim}dim====================================")
                
                
                psnr_fram1000_txt = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}/ckpt/quan_exp_dim_psnr_results_frame500.txt"
                # 读取psnr_fram1000_txt最后一行，格式为Max PSNR: 34.281218 at checkpoint 135000，找到最佳的ckpt
                logname =f"log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}"
                
                
                try:
                    with open(psnr_fram1000_txt, "r") as f:
                        lines = f.readlines()
                        print("split :",lines[-1].split(" ") )
                        best_chkpnt = lines[-1].split(" ")[-2]
                        print("find best checkpoint:", best_chkpnt)
                except:
                    print(f"Error reading {psnr_fram1000_txt}")
                video1_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/{logname}/rec_{quan_bit}bit_exp_dim{exp_dim}_{best_chkpnt}_frame1000_best_compressed.avi"
                video2_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/{logname}/ori_frame1000.avi"
                
                #  获得测试集上的结果          
                command = f'CUDA_VISIBLE_DEVICES="3" /home/ljl/workspace/ICME2025/LIC_TCM/tcm_venv/bin/python train_smirk_eagles.py \
                --config configs/efficient-3dgs.yaml \
                --idname {id_name}_smirk --exp_dim {exp_dim} \
                --logname log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)} \
                --quan_bit {quan_bit} --loss_type lpips --skip_train --iteration {best_chkpnt} --mlp_layer 6 \
                --chkpnt_number {best_chkpnt} --compress_test'
                            
                print("command",command)
                process = subprocess.run(command, shell=True)
                if process.returncode != 0:
                    print(f"Error running {script_name} for checkpoint {id_name}, {quan_bit}, {exp_dim}")
                    continue
                else:
                    print(f"Success running {script_name} for checkpoint {id_name}, {quan_bit}, {exp_dim}")


import gzip              
def calculate_compressed_model_size():
    id_list = ["id2","id4","id8","Obama","id5"]
    quan_bit_list = [8,10]
    exp_dim_list = [10,20,30,40,50]
    
    # id_list = ["id2"]
    # quan_bit_list = [10]
    # exp_dim_list = [10]

    script_name = "train_smirk_eagles.py"

    total_mlp_size = 0
    total_gaussian_size = 0
    total_mlp_size_gz = 0
    total_gaussian_size_gz = 0 
    
    total_gaussian_size_best_fp16 = 0
    total_gaussian_size_best_fp16_gz = 0
    
    total_num = 0
    results = [] # 保存idname, quan_bit, exp_dim, chkpnt, psnr
    for id_name in id_list:
        for quan_bit in quan_bit_list:
            for exp_dim in exp_dim_list:
            
                print(f"==============================start_{id_name}_smirk_{quan_bit}bit_{exp_dim}dim====================================")
                
                
                psnr_fram1000_txt = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}/ckpt/quan_exp_dim_psnr_results.txt"
                # 读取psnr_fram1000_txt最后一行，格式为Max PSNR: 34.281218 at checkpoint 135000，找到最佳的ckpt
                logname =f"log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}"
                
                
                try:
                    with open(psnr_fram1000_txt, "r") as f:
                        lines = f.readlines()
                        print("split :",lines[-1].split(" ") )
                        best_chkpnt = lines[-1].split(" ")[-2]
                        print("find best checkpoint:", best_chkpnt)
                except:
                    print(f"Error reading {psnr_fram1000_txt}")
                # video1_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/{logname}/rec_{quan_bit}bit_exp_dim{exp_dim}_{best_chkpnt}_frame1000_best_compressed.avi"
                # video2_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/{logname}/ori_frame1000.avi"
                
                # ckpt_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/{logname}/ckpt/chkpnt{best_chkpnt}_mlp.pth" 
                gaussian_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/{logname}/ckpt/point_cloud_compressed{best_chkpnt}.pkl"
                gaussian_path_best_fp16 = gaussian_path.replace(".pkl", "_best_fp16.pkl")
                gaussian_path_best_fp16_gz = gaussian_path_best_fp16 + ".gz"
                #  获得测试集上的结果 
                
                # 使用gzip分别对这两个文件进行压缩，并保留压缩前原文件，压缩的路径为ckpt_path.gz和gaussian_path.gz
                # ckpt_path_gz = ckpt_path + ".gz"
                # gaussian_path_gz = gaussian_path + ".gz"
                with open(gaussian_path_best_fp16_gz, "wb") as f:
                    with gzip.open(f, "wb") as f:
                        f.write(open(gaussian_path_best_fp16, "rb").read())
                # with open(gaussian_path_best_fp16_gz, "wb") as f:
                #     with gzip.open(f, "wb") as f:
                #         f.write(open(gaussian_path_best_fp16, "rb").read())
                # 计算两个pth文件分别占用多少存储空间
                # ckpt_size = os.path.getsize(ckpt_path)
                gaussian_size = os.path.getsize(gaussian_path)
                gaussian_size_best_fp16 = os.path.getsize(gaussian_path_best_fp16)
                gaussian_size_best_fp16_gz = os.path.getsize(gaussian_path_best_fp16_gz)
                
                # ckpt_size_gz = os.path.getsize(ckpt_path_gz)
                # gaussian_size_gz = os.path.getsize(gaussian_path_gz)
                
                # print(f"ckpt_size: {ckpt_size} bytes")
                # print(f"gaussian_size: {gaussian_size} bytes")
                # print(f"ckpt_size_gz: {ckpt_size_gz} bytes")
                # print(f"gaussian_size_gz: {gaussian_size_gz} bytes")
                
                # total_mlp_size += ckpt_size
                total_gaussian_size += gaussian_size
                total_gaussian_size_best_fp16 += gaussian_size_best_fp16
                total_gaussian_size_best_fp16_gz += gaussian_size_best_fp16_gz
                # total_mlp_size_gz += ckpt_size_gz
                # total_gaussian_size_gz += gaussian_size_gz
                
                total_num += 1  
    
    
    # print(f"mlp_size_per_num: {(total_mlp_size/total_num /1024):.2f} K bytes")
    print(f"gaussian_size_per_num: {(total_gaussian_size/total_num/1024):.2f} K bytes")
    print(f"gaussian_size_best_fp16_per_num: {(total_gaussian_size_best_fp16/total_num/1024):.2f} K bytes")
    # print(f"mlp_size_gz_per_num: {(total_mlp_size_gz/total_num/1024):.2f} K bytes")
    print(f"gaussian_size_best_fp16_gz_per_num: {(total_gaussian_size_best_fp16_gz/total_num/1024):.2f} K bytes")
         
def test_smirk_eagles():
    id_name_list_smirk = ["Obama_smirk"]
    quan_bit_list = [8,10]
    # quan_bit_list = [0]
    exp_dim_list = [10,20,30,40,50]
    # exp_dim_list = [50]
    # id_name_list_emoca = ["Obama_emoca","id4_emoca", "id5_emoca","id7_emoca", "id8_emoca","ljl_wo_glass_emoca"]
    # id_name_list= ["Obama"]
    mlp_num = 6

    script_name = "train_smirk_eagles.py"
    for idname in id_name_list_smirk:
        print("idname",idname)
        for quan_bit in quan_bit_list:
            
            for exp_dim in exp_dim_list:
                results = []
                
                # logname =f"log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}" # 40剪枝
                logname =f"log_smirk_eagles_mlp6_rest141_purn0_quan{quan_bit}_exp_dim_{str(exp_dim)}"
                
                base_checkpoint_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{idname}/{logname}/ckpt/"
                
                chkpnts = list(range(70000, 150001, 10000))  # 生成原始范围列表
                # chkpnts = []
                chkpnts.append(135000)                      # 添加 135000
                # chkpnts = list(range(150000, 200001, 10000))
                chkpnts = sorted(chkpnts)                   # 排序
                # 循环从 25000 到 150000，步长 5000
                for chkpnt in chkpnts:
                    video1_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{idname}/{logname}/rec_{quan_bit}bit_exp_dim{exp_dim}_{chkpnt}_frame1000.avi"
                    video2_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{idname}/{logname}/ori_frame1000.avi"
                    # video1_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{idname}/{logname}/rec.avi"
                    # video2_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{idname}/{logname}/ori.avi"
                    checkpoint_file = os.path.join(base_checkpoint_path, f"chkpnt{chkpnt}.pth")
                    print(f"Running checkpoint: {checkpoint_file}")

                    # 运行 test_smirk.py
                    # command = f'CUDA_VISIBLE_DEVICES="3" /home/ljl/workspace/ICME2025/LIC_TCM/tcm_venv/bin/python {script_name} \
                    #             --config configs/efficient-3dgs.yaml \
                    #             --idname {idname} \
                    #             --exp_dim {exp_dim} \
                    #             --logname {logname} \
                    #             --quan_bit 0 \
                    #             --loss_type lpips \
                    #             --skip_train \
                    #             --iteration {chkpnt} \
                    #             --mlp_layer {mlp_num}'
                                
                    command = f'CUDA_VISIBLE_DEVICES="3" /home/ljl/workspace/ICME2025/LIC_TCM/tcm_venv/bin/python train_smirk_eagles.py \
                    --config configs/efficient-3dgs.yaml \
                    --idname {idname} --exp_dim {exp_dim} \
                    --logname {logname} \
                    --quan_bit {quan_bit} --loss_type lpips --skip_train --iteration {chkpnt} --mlp_layer 6 \
                    --chkpnt_number {chkpnt}'
                                
                    print("command",command)
                    process = subprocess.run(command, shell=True)
                    if process.returncode != 0:
                        print(f"Error running {script_name} for checkpoint {checkpoint_file}")
                        continue
                    else:
                        print(f"Success running {script_name} for checkpoint {checkpoint_file}")

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
                output_file = os.path.dirname(base_checkpoint_path) + f"/quan_exp_dim_psnr_results.txt"
                # output_file = "psnr_results.txt"
                with open(output_file, "a") as f:
                    for idname,quan_bit, exp_dim, chkpnt, psnr in results:
                        f.write(f"{idname}, {quan_bit}, {chkpnt}, {psnr}\n")
                    # 写入最大psnr以及对应的checkpoint
                    f.write(f"Max PSNR: {max_psnr[4]} at checkpoint {max_psnr[3]} \n")

                print(f"Results saved to {output_file}")
                
         
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, hidden_layers=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fcs = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_layers-1)]
        )
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        # input: B,V,d
        batch_size, N_v, input_dim = input.shape
        input_ori = input.reshape(batch_size*N_v, -1)
        h = input_ori
        for i, l in enumerate(self.fcs):
            h = self.fcs[i](h)
            h = F.relu(h)
        output = self.output_linear(h)
        output = output.reshape(batch_size, N_v, -1)

        return output  
    
def save_pruned_model_with_mask_fp16(model, save_path):
    """ 记录掩码并保存剪枝后的 MLP，减少存储空间 """
    
    weight_masks = {}
    weight_values = {}
    biases = {}

    for name, param in model.state_dict().items():
        if "weight" in name:  # 只对权重进行掩码存储
            mask = (param != 0).cpu().numpy().astype(np.uint8)  # 转换为 uint8 以便打包
            packed_mask = np.packbits(mask)  # 1-bit 压缩
            weight_masks[name] = packed_mask
            weight_values[name] = param[param != 0].half()  # 仅存储非零权重，并转换为 fp16
        else:
            biases[name] = param.half()  # 直接转换为 fp16 并存储偏置参数
    
    # # 分别保存 mask、权重值和偏置
    # torch.save(weight_masks, f"{save_path}_masks_fp16.pt")      # 存储 1-bit mask
    # torch.save(weight_values, f"{save_path}_values_fp16.pt")    # 存储 fp16 非零权重
    # torch.save(biases, f"{save_path}_biases_fp16.pt")           # 存储 fp16 偏置
    
    # # 兼容性存储完整剪枝后的 state_dict
    # pruned_state_dict = {
    #     **{f"{k}_mask": v for k, v in weight_masks.items()},
    #     **{f"{k}_values": v for k, v in weight_values.items()},
    #     **biases
    # }
    
    torch.save({"masks": weight_masks, "values": weight_values, "biases": biases}, save_path)
    # torch.save(pruned_state_dict, save_path+"_fp16.pth")
    
def calculate_compressed_mlp_size():
    id_list = ["id2","id4","id8","Obama","id5"]
    quan_bit_list = [8,10]
    exp_dim_list = [10,20,30,40,50]
    

    total_mlp_size = 0
    total_mlp_size_gz = 0
    total_num = 0
  
    list_less_10k = []
    for id_name in id_list:
        for quan_bit in quan_bit_list:
            for exp_dim in exp_dim_list:
            
                print(f"==============================start_{id_name}_smirk_{quan_bit}bit_{exp_dim}dim====================================")
                
                psnr_fram1000_txt = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}/ckpt/quan_exp_dim_psnr_results.txt"
                # 读取psnr_fram1000_txt最后一行，格式为Max PSNR: 34.281218 at checkpoint 135000，找到最佳的ckpt
                logname =f"log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}"
                
                try:
                    with open(psnr_fram1000_txt, "r") as f:
                        lines = f.readlines()
                        print("split :",lines[-1].split(" ") )
                        best_chkpnt = lines[-1].split(" ")[-2]
                        print("find best checkpoint:", best_chkpnt)
                except:
                    print(f"Error reading {psnr_fram1000_txt}")
                # video1_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/{logname}/rec_{quan_bit}bit_exp_dim{exp_dim}_{best_chkpnt}_frame1000_best_compressed.avi"
                # video2_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/{logname}/ori_frame1000.avi"
                
                ckpt_path = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/{logname}/ckpt/chkpnt{best_chkpnt}_mlp.pth" 
                
                #  获得测试集上的结果 
                
                # mlp = MLP(input_dim=51+55, output_dim=10, hidden_dim=256, hidden_layers=6)
                # mlp_checkpoint = torch.load(ckpt_path)

                # # 解析模型参数
                # if len(mlp_checkpoint) == 3:
                #     (model_params, first_iter, best_psnr) = mlp_checkpoint
                # elif len(mlp_checkpoint) == 2:
                #     (model_params, first_iter) = mlp_checkpoint
                # else:
                #     model_params = mlp_checkpoint

                # (net_params, opt_params) = model_params
                # mlp.load_state_dict(net_params)

                # 保存剪枝后的模型
                pruned_path = ckpt_path.replace(".pth","best_compressed_fp16.pth") 
                
                # save_pruned_model_with_mask_fp16(mlp, pruned_path)
                
                
                # 使用gzip分别对这两个文件进行压缩，并保留压缩前原文件，压缩的路径为ckpt_path.gz和gaussian_path.gz
                ckpt_path_gz = pruned_path + ".gz"
                # gaussian_path_gz = gaussian_path + ".gz"
                # with open(ckpt_path_gz, "wb") as f:
                #     with gzip.open(f, "wb") as f:
                #         f.write(open(pruned_path, "rb").read())
                
                # 计算两个pth文件分别占用多少存储空间
                ckpt_size = os.path.getsize(pruned_path)
                ckpt_size_gz = os.path.getsize(ckpt_path_gz)
                
                print(f"ckpt_size of {id_name} {quan_bit}bit {exp_dim}dim: {ckpt_size} bytes")
                print(f"ckpt_size_gz of {id_name} {quan_bit}bit {exp_dim}dim: {ckpt_size_gz} bytes")
                
                if int(best_chkpnt) > 100000:
                    total_mlp_size += ckpt_size
                    total_mlp_size_gz += ckpt_size_gz
                
                    total_num += 1  
                else:
                    list_less_10k.append((id_name, quan_bit, exp_dim, best_chkpnt))
    
    print(f"mlp_size_per_num: {(total_mlp_size/total_num /1024):.2f} K bytes")
    
    print(f"mlp_size_gz_per_num: {(total_mlp_size_gz/total_num/1024):.2f} K bytes")
    print(f"list_less_10k: {list_less_10k}")
                            
def id_ssim_lpips_mannul():
    # id_list = ["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8"]
   
    test_list = [('id2', 10, 10, '90000'), ('id2', 10, 50, '90000'), ('id4', 8, 20, '90000'), ('id4', 8, 30, '90000'), \
                 ('id4', 8, 50, '90000'), ('id4', 10, 20, '80000'), ('id4', 10, 30, '80000'), ('id8', 10, 20, '90000'),\
                 ('id8', 10, 40, '90000'), ('id5', 10, 40, '90000')]
    
    for id_name, quan_bit, exp_dim, chkpnt in test_list:
    
    
        print(f"==============================start_{id_name}_smirk_{quan_bit}bit_{exp_dim}dim====================================")
        psnr_fram1000_txt = f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}/ckpt/quan_exp_dim_psnr_results.txt"
        
        # 读取psnr_fram1000_txt最后一行，格式为Max PSNR: 34.281218 at checkpoint 135000，找到最佳的ckpt
        try:
            with open(psnr_fram1000_txt, "r") as f:
                lines = f.readlines()
                print("split :",lines[-1].split(" ") )
                best_chkpnt = lines[-1].split(" ")[-2]
                print("find best checkpoint:", best_chkpnt)
        except:
            print(f"Error reading {psnr_fram1000_txt}")
            
        # 计算SSIM和LPIPS
        best_rec_video = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}", f"rec_{quan_bit}bit_exp_dim{exp_dim}_{best_chkpnt}_frame1000.avi")
        ori_video_path = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}", f"ori_frame1000.avi")
        print("best_rec_video:", best_rec_video)
        print("ori_video_path:", ori_video_path)
        
        # 使用ffmpeg将两个avi视频转成图片序列
        best_rec_imgs_path = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}", "rec_best_imgs_mannul")
        ori_imgs_path = os.path.join(f"/home/ljl/workspace/IBC24/FlashAvatar-code/dataset/{id_name}_smirk/log_smirk_eagles_mlp6_rest141_purn40_quan{quan_bit}_exp_dim_{str(exp_dim)}", "ori_imgs")
        
        os.makedirs(best_rec_imgs_path, exist_ok=True)
        os.makedirs(ori_imgs_path, exist_ok=True)
        os.system(f"ffmpeg -i {best_rec_video} -vf \"fps=30\" -q:v 2 -start_number 1 {best_rec_imgs_path}/%05d.png")
        # os.system(f"ffmpeg -i {ori_video_path} -vf \"fps=30\" -q:v 2 -start_number 1 {ori_imgs_path}/%05d.png")
        
        # ffmpeg -i rec_10bit_exp_dim30_150000_frame1000.avi -vf "fps=25" rec_%05d.png
        
        # 计算SSIM和LPIPS
        ssim_scores = []
        lpips_scores = []
        for i in range(1,1001):
            pred_image = os.path.join(best_rec_imgs_path, f"{i:05}.png")
            gt_image = os.path.join(ori_imgs_path, f"{i:05d}.png")
            ssim_score = calculate_ssim(gt_image, pred_image)
            ssim_scores.append(ssim_score)
            lpips_score = calculate_lpips(gt_image, pred_image)
            lpips_scores.append(lpips_score)
        avg_ssim = round(sum(ssim_scores)/len(ssim_scores),4)
        avg_lpips = round(sum(lpips_scores)/len(lpips_scores),4)
        print(f"==============================result_of_{id_name}_smirk_{quan_bit}bit_{exp_dim}dim================================================")
        
        print(f"{id_name}_smirk, quan_bit {quan_bit}, exp_dim {exp_dim} SSIM scores:", avg_ssim)
        print(f"{id_name}_smirk, quan_bit {quan_bit}, exp_dim {exp_dim} LPIPS scores:", avg_lpips)
        write_gs_ssim_lpips_to_file(id_name, quan_bit, exp_dim, avg_ssim, avg_lpips)
                
if __name__ == "__main__":
    # get_265gt()
    # rgb_yuv()
    # yuv_265()
    # hevc_ssim_lpips()
    # delete_yuv()
    # test_smirk_eagles()
    id_ssim_lpips_purn0()
    # id_ssim_lpips()
    # test_smirk_eagles_compressed_best()
    # get_compressed_model()
    # calculate_compressed_model_size()
    # calculate_compressed_mlp_size()
    # id_ssim_lpips_mannul()