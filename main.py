import numpy as np
import torch
from keras.backend import one_hot
from torch import nn,optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

#训练集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=False)

#测试集
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=False)

"""
    root:程序存放位置， ./data为当前存放位置
    train：表示载入的为训练集的数据
    transform= 将载入进来的数据转化为pytorch框架中的基本类型tensor
    首次下载后可改成 download=False，避免误删重下。
    如果想换位置，把 root 换成别的路径即可，例如 D:/datasets.
"""
#批次大小
batch_size = 64

#装载数据集
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


#定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784,10)#此处要求输入是二维
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        #将输入展平成（batch_size，784）
        x = x.view(x.size(0), -1)#x.size(0)保持batch维度，-1让pytorch自动计算剩下的维度，这里是1*28*28 =784
        x = self.fc1(x)
        x = self.softmax(x)
        return x

LR = 0.5#学习率
#定义模型
model = Net()
#定义损失函数
mse_loss = nn.MSELoss()
#定义优化器
optimizer = optim.SGD(model.parameters(), lr=LR)

def train():
    for i,data in enumerate(train_loader):
        # 获得一个批次的数据
        inputs, labels = data
        #获得模型预测结果（64，10）
        out = model(inputs)
        #to onehot，把数据标签变成独热编码
        labels = labels.reshape(-1,1)   #（64）- (64,1)
        #tensor.scatter(dim,index,src)
        #dim:对哪个维度进行独热编码
        #index：要将src中对应的值放到tensor的哪个位置。
        #src：插入index的数值
        one_hot = torch.zeros(inputs.size(0),10)
        one_hot = one_hot.scatter(1,labels,1)
        #计算loss，mes_loss的两个数据的shape要一致
        loss = mse_loss(out,one_hot)
        #梯度清零
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        #修改权值
        optimizer.step()

def test():
    correct = 0
    total = 0
    for i,data in enumerate(test_loader):
        # 获得一个批次的数据和标签
        inputs, labels = data
        # 获得模型预测结果（64，10）
        out = model(inputs)
        # 找出每一行概率最大的类别
        _, predicted = torch.max(out.data, 1)
        #预测统计对了多少个
        correct += (predicted == labels).sum().item()
        total += labels.size(0)#累计测试样本总数
    print("test acc: {0}".format(correct / total))#模型在测试集上的分类准确率

if __name__ == "__main__":
    #训练循环放在主函数中
    for epoch in range(10):
        print("epoch:{}".format(epoch))
        train()
        test()