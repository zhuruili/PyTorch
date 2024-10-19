# env: Python 3.9.18
# 简单线性回归示例
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

# 超参数设置
input_size = 1
output_size = 1
num_epochs = 500
learning_rate = 0.001
save_model = False # 是否保存模型
SAVEPATH = 'Rookie\part_2_items\MachineLearning\PyTorch\model\LR_simple.ckpt' # 模型保存路径

# 数据集
x_train = torch.tensor([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                        [9.779], [6.182], [7.59], [2.167], [7.042], 
                        [10.791], [5.313], [7.997], [3.1]], dtype=torch.float32)

y_train = torch.tensor([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                        [3.366], [2.596], [2.53], [1.221], [2.827], 
                        [3.465], [1.65], [2.904], [1.3]], dtype=torch.float32)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Linear regression model
model = nn.Linear(input_size, output_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  

# Move tensors to the configured device
x_train = x_train.to(device)
y_train = y_train.to(device)

# Train the model
print('--- Start Training ---')
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
print('--- End Training ---')

# Plot the graph
predicted = model(x_train).detach().cpu().numpy()
plt.plot(x_train.cpu().numpy(), y_train.cpu().numpy(), 'ro', label='Original data')
plt.plot(x_train.cpu().numpy(), predicted, label='Fitted line')
plt.legend()
plt.show()

# Save the model checkpoint
if save_model:
    torch.save(model.state_dict(), SAVEPATH)
    print(f'Model saved to {SAVEPATH}')
else:
    print('save_model: False')



"""
这段Python代码使用了PyTorch库来进行简单的线性回归训练。

首先，你导入了PyTorch及其相关的模块：

torch：PyTorch的核心库。
torch.nn：PyTorch的神经网络库，用于构建神经网络模型。
torch.optim：优化算法库，用于训练模型。
接着，你定义了超参数：

input_size 和 output_size：分别定义了输入和输出的特征数量，这里都是1，因为我们的数据只有一个特征。
num_epochs：训练的轮数，也就是模型会遍历多少次整个数据集。
learning_rate：学习率，决定了模型在训练过程中参数更新的步长。
然后，你创建了一个数据集：

x_train：输入数据，是一个15x1的张量（可以理解为矩阵），包含15个样本，每个样本有一个特征。
y_train：对应的目标值，也是一个15x1的张量，包含了每个样本对应的真实输出。
接下来，你定义了一个线性回归模型：

使用nn.Linear创建了一个线性层，输入大小为1，输出大小也为1。
随后，你定义了损失函数和优化器：

损失函数criterion使用了均方误差损失（MSE），它计算了模型输出与目标值之间的平均平方误差。
优化器optimizer使用了随机梯度下降（SGD），它会根据损失函数的梯度来更新模型的参数。
然后，你开始了模型的训练过程：

在每一个epoch中，模型会首先进行前向传播，计算输出并计算损失。
接着，模型会进行反向传播，计算损失关于模型参数的梯度。
然后，使用优化器来更新模型的参数。
每5个epoch，代码会打印当前的epoch数和损失值，以便观察训练过程。
最后，你保存了模型的参数：

使用torch.save将模型的state_dict（包含了模型的所有参数）保存到了一个文件中，这样以后就可以重新加载这个模型而不需要重新训练。
简而言之，这段代码使用PyTorch库创建了一个简单的线性回归模型，并使用给定的玩具数据集进行训练，最后保存了训练好的模型参数。
"""