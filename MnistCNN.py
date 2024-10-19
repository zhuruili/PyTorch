"""
搭建一个简单的CNN模型，用于对Mnist数据集进行处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore') # 忽略警告

# 超参数设置
input_size = 28 #图像的总尺寸28*28
num_classes = 10 #类别数
num_epochs = 4 #训练次数
batch_size = 64 #批处理大小
learning_rate = 0.001 #学习率
DOWNLOAD_MNIST = False  # 是否下载数据集，如果数据集下载好了就写False

# 训练集
train_data = datasets.MNIST(
    root='Rookie\part_2_items\MachineLearning\PyTorch\data',  # 下载文件的保存路径或数据提取路径
    train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
    transform=transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray

    download=DOWNLOAD_MNIST,  # 已经下载了就不需要下载了
)

# 测试集
test_data = torchvision.datasets.MNIST(
    root='Rookie\part_2_items\MachineLearning\PyTorch\data',
    train=False,  # 表明是测试集
    transform=transforms.ToTensor()
)

# 数据加载
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                            batch_size=batch_size,
                                            shuffle=False)

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,      # 输入特征图个数
                out_channels=16,    # 输出特征图个数（卷积核个数）
                kernel_size=5,      # 卷积核大小
                stride=1,           # 步长
                padding=2           # 边缘填充
            ),
            nn.ReLU(),              # 激活函数
            nn.MaxPool2d(kernel_size=2)  # 池化层
        )
        self.layer2 = nn.Sequential(    # 第二层
            nn.Conv2d(16,32,5,1,2),     # 输入通道数16，输出通道数32
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32*7*7,num_classes)  # 全连接层

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0),-1)  # 展平
        output = self.out(x)
        return output
    
def CheckAcc(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)

# 实例化模型
model = CNN()
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

print(model)
print('Train data size: {}'.format(len(train_data)))
print('Test data size: {}'.format(len(test_data)))
print("------------------Start Training------------------")

# 训练模型
for epoch in range(num_epochs):
    train_rights = []  # 记录当前epoch的结果
    for batch_idx, (data, target) in enumerate(train_loader):  # 从train_loader中取出一个批次的数据进行训练
        model.train()  # 训练模式
        output = model(data)  # 前向传播
        loss = criterion(output, target)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        right = CheckAcc(output, target)  # 计算正确率
        train_rights.append(right)  # 记录结果
        
        if batch_idx % 100 == 0:  # 每100个batch打印一次结果
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    # 计算当前epoch的正确率
    right_ratio = 1.0 * np.sum([i[0] for i in train_rights]) / np.sum([i[1] for i in train_rights])
    print('Train set: Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        epoch, loss.item(), np.sum([i[0] for i in train_rights]), np.sum([i[1] for i in train_rights]),
        100. * right_ratio))

print("------------------Training End------------------")
        
