import torch
import numpy as np

# 定义概率分布
probs = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1])  # 总和应为 1

# 根据概率生成随机数
number = np.zeros(5)
print(number)
for i in range(10000):
    n = torch.multinomial(probs, 1).item()  # 生成一个随机数
    number[n] += 1
    # print("Generated number:", n)
print("Generated number:", number)
