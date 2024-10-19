"""
搭建PyTorch神经网络进行气温预测
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn

import datetime
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")

# 超参数设置
data_path = 'Rookie\part_2_items\MachineLearning\PyTorch\data\\temperature.csv'  # 数据路径
save_path = 'Rookie\part_2_items\MachineLearning\PyTorch\data\\temp_model.pth'  # 模型保存路径


features = pd.read_csv(data_path) # 读取数据
print(features.head())
print(features.shape)

# 处理时间数据
years = features['year']
months = features['month']
days = features['day']
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
print(dates[:5])

# 绘制气温变化图
plt.style.use('fivethirtyeight')  # 设置绘图风格
fig,((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))  # 创建画布
fig.autofmt_xdate(rotation=45)  # 旋转日期标签
# 真实气温
ax1.plot(dates, features['actual'])
ax1.set_xlabel(''); ax1.set_ylabel('Temperature'); ax1.set_title('Max Temp')
# 前一天气温
ax2.plot(dates, features['temp_1'])
ax2.set_xlabel(''); ax2.set_ylabel('Temperature'); ax2.set_title('Previous Max Temp')
# 前两天气温
ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date'); ax3.set_ylabel('Temperature'); ax3.set_title('Two Days Prior Max Temp')
# 来自逗逼朋友的预测
ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date'); ax4.set_ylabel('Temperature'); ax4.set_title('Friend Estimate')
plt.tight_layout(pad=2)  # 调整子图之间的间距
plt.show()

# 对week变量进行独热编码
features = pd.get_dummies(features)  # 对week进行独热编码（它会自动判断谁需要编码）
print(features.head())

# 提取标签
labels = features['actual']  # 提取真实气温
features = features.drop('actual', axis=1)  # 从特征中删除真实气温
features_list = list(features.columns)  # 保存特征名称以备后用
features = np.array(features)  # 转换为数组
print(features.shape)

# 标准化数据
input_features = preprocessing.StandardScaler().fit_transform(features)  # 标准化数据
print(input_features[0])

# 将数据转换为张量
x = torch.tensor(input_features, dtype=float)  # 特征张量
y = torch.tensor(labels.values, dtype=float)  # 标签张量
print(x[:5])
print(y[:5])

# 权重及参数初始化
weights = torch.randn((14,128), dtype=float, requires_grad=True)  # 权重张量
biases = torch.randn(128, dtype=float, requires_grad=True)  # 偏置张量
weights2 = torch.randn((128,1), dtype=float, requires_grad=True)  # 权重张量
biases2 = torch.randn(1, dtype=float, requires_grad=True)  # 偏置张量

learning_rate = 0.001  # 学习率
losses = []  # 保存损失值
epochs = 1000  # 迭代次数

# 训练模型
for epoch in range(epochs):
    # 前向传播
    hidden = torch.matmul(x, weights) + biases  # 计算隐藏层
    hidden = torch.relu(hidden)  # 使用ReLU激活函数
    predictions = torch.matmul(hidden, weights2) + biases2  # 计算预测值
    
    # 计算损失
    loss = torch.mean((predictions - y)**2)
    losses.append(loss.data.numpy())  # 保存损失值

    # 反向传播
    loss.backward()
    
    # 更新权重和偏置
    weights.data.add_(-learning_rate * weights.grad.data)
    biases.data.add_(-learning_rate * biases.grad.data)
    weights2.data.add_(-learning_rate * weights2.grad.data)
    biases2.data.add_(-learning_rate * biases2.grad.data)
    
    # 清空梯度
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()
    
    if epoch % 100 == 0:
        print('Epoch %d, Loss: %f' % (epoch, loss.data.numpy()))

"""
以上模型训练方法是最基础（而且每一步都是手写的）的，实际上PyTorch提供了更高级的API，可以更方便地构建和训练神经网络模型。
"""

# 使用PyTorch简化模型训练
input_size = input_features.shape[1]  # 输入大小
hidden_size = 128  # 隐藏层大小
output_size = 1  # 输出层大小
batch_size = 16  # 批量大小

# 使用Sequential定义模型
model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.Sigmoid(),
    nn.Linear(hidden_size, output_size)
)

# 定义损失函数和优化器
criterion = nn.MSELoss(reduction='mean')  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器

# 训练模型
losses = []
for epoch in range(epochs):
    batch_loss = []
    # MINI-Batch训练
    for start in range(0, len(x), batch_size):
        end = start + batch_size if start + batch_size < len(x) else len(x)
        xx = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
        yy = torch.tensor((labels.values)[start:end], dtype=torch.float, requires_grad=True)
        
        # 前向传播
        output = model(xx)
        loss = criterion(output, yy.view(-1, 1))
        batch_loss.append(loss.data.numpy())
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
    # 打印损失
    if epoch % 100 == 0:
        losses.append(np.mean(batch_loss))
        print('Epoch %d, Loss: %f' % (epoch, np.mean(batch_loss)))

# 测试模型
x = torch.tensor(input_features, dtype=torch.float)
predictions = model(x).data.numpy()

true_data = pd.DataFrame(data={'date': dates, 'actual': labels})  # 用表格保存日期和真实数据
predictions_data = pd.DataFrame(data={'date': dates, 'predictions': predictions.reshape(-1)})  # 用表格保存日期和预测数据

# 绘制真实数据和预测数据
plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')
plt.plot(predictions_data['date'], predictions_data['predictions'], 'r-', label='predictions')
plt.xticks(rotation=60)
plt.legend()
plt.xlabel('Date'); plt.ylabel('Max Temperature (F)'); plt.title('Actual and Predicted Values')
plt.show()