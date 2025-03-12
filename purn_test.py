import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义简单的 MLP 模型
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def prune(self, pruning_ratio):
        """
        对模型的权重进行剪枝
        :param pruning_ratio: 剪枝比例 (0 到 1)
        """
        with torch.no_grad():
            for layer in [self.fc1, self.fc2]:
                # 获取权重绝对值排序
                weight = layer.weight.data
                threshold = np.percentile(torch.abs(weight).cpu().numpy(), pruning_ratio * 100)
                
                # 创建 mask，低于阈值的权重置为 0
                mask = torch.abs(weight) > threshold
                layer.weight.data *= mask  # 应用剪枝
                layer.weight_mask = mask   # 保存 mask，用于追踪剪枝状态
                # 冻结被剪枝的权重
                layer.weight.requires_grad = False
                
    # def prune_and_freeze(self, pruning_ratio):
    #     for name, param in self.named_parameters():
    #         if 'weight' in name:
    #             # 获取权重的绝对值排序索引
    #             num_to_prune = int(pruning_ratio * param.numel())
    #             if num_to_prune > 0:
    #                 threshold = torch.topk(param.abs().flatten(), num_to_prune, largest=False).values[-1]
    #                 mask = (param.abs() > threshold).float()  # 剪枝的 mask
    #                 param.data *= mask  # 将被剪枝的权重置为 0
                    
    #                 # 将 mask 注册为模型的 buffer
    #                 self.register_buffer(f"{name}_pruned_mask", mask)

    #                 # 冻结剪枝后的权重
    #                 param.requires_grad = False  # 防止这些权重被优化

                    
                    
    # 在反向传播后确保剪枝权重不会被更新
    def zero_out_pruned_weights(self):
        for name, param in self.named_parameters():
            if 'pruned_mask' in param._buffers:
                mask = param._buffers['pruned_mask']
                param.grad = param.grad * mask  # 保证被剪枝的权重梯度为 0


# 查看剪枝之后模型权重中0的比例

def zaro_ratio(model):
    # 查看剪枝后的模型信息
    total_params = 0
    total_zero_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        num_zero_params = (param == 0).sum().item()
        zero_ratio = num_zero_params / num_params

        total_params += num_params
        total_zero_params += num_zero_params

        print(f"{name}, Pruned: {num_zero_params} / {num_params} ({zero_ratio:.2%})")

    # 查看整个模型权重为 0 的比例
    overall_zero_ratio = total_zero_params / total_params
    print(f"\nOverall Pruned Ratio: {total_zero_params} / {total_params} ({overall_zero_ratio:.2%})")
    
# 数据生成
def generate_data(num_samples=1000, input_size=10, output_size=1):
    X = torch.rand(num_samples, input_size)
    y = X.sum(dim=1, keepdim=True) + torch.rand(num_samples, output_size) * 0.1
    return X, y

# 设置参数
input_size = 10
hidden_size = 32
output_size = 1
initial_epochs = 100
fine_tune_epochs = 40
prune_ratio = 0.3

# 初始化模型、优化器和损失函数
model = SimpleMLP(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 生成训练数据
X_train, y_train = generate_data()

# 阶段 1：初始训练
print("Stage 1: Initial Training")
for epoch in range(initial_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{initial_epochs}], Loss: {loss.item():.4f}")

# 剪枝模型
print("\nStage 2: Pruning")
model.prune(pruning_ratio=prune_ratio)

# 查看剪枝后的模型信息
for name, param in model.named_parameters():
    print(f"{name}, Pruned: {(param == 0).sum()} / {param.numel()}")

zaro_ratio(model)


# 阶段 3：微调
print("\nStage 3: Fine-tuning")
for epoch in range(fine_tune_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    # model.zero_out_pruned_weights()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{fine_tune_epochs}], Fine-tuning Loss: {loss.item():.4f}")

zaro_ratio(model)
# 测试微调后的模型性能
X_test, y_test = generate_data(num_samples=100)
with torch.no_grad():
    y_pred = model(X_test)
    print("\nFinal Test MSE Loss:", criterion(y_pred, y_test).item())
