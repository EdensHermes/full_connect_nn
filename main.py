"""
MNIST 手写数字识别 —— 全连接神经网络（优化版）

相比最初的 MSE 版本，主要改进：
  1. 损失函数：MSELoss + Softmax → CrossEntropyLoss（梯度更大、收敛更快，也不再需要 one-hot）
  2. 网络结构：单层 784→10 → 多个隐藏层，每层带 BatchNorm + ReLU + Dropout
  3. 数据处理：输入标准化 + 训练时随机旋转/缩放/平移（数据增强，减少过拟合）
  4. 训练策略：AdamW 优化器 + OneCycle 学习率 + 标签平滑
  5. 工程细节：自动使用 GPU、数据增强在 GPU 上整批完成、固定随机种子、打印 loss/准确率/耗时、保存模型
"""
import math
import time

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

# ================================ 超参数 ================================
dataRoot = './data'               # 数据集存放位置，想换位置可改成例如 'D:/datasets'
hiddenSizes = [1024, 512, 256]    # 每个隐藏层的神经元个数
batchSize = 128                   # 批次大小
epochNum = 30                     # 训练轮数
maxLearningRate = 3e-3            # OneCycle 策略中学习率的峰值
weightDecay = 1e-4                # 权重衰减（类似 L2 正则），抑制过拟合
dropoutRate = 0.3                 # Dropout 比例：训练时每层随机"关掉" 30% 的神经元
labelSmoothing = 0.1              # 标签平滑：把目标概率 1.0 软化成 0.9，防止模型过度自信
maxRotateDegrees = 10             # 数据增强：最大旋转角度（度）
maxTranslate = 0.1                # 数据增强：最大平移比例（相对图片边长）
scaleRange = (0.9, 1.1)           # 数据增强：随机缩放倍数范围
randomSeed = 0                    # 随机种子，保证每次运行结果基本一致
modelPath = 'mnist_fc.pt'         # 模型保存路径

# MNIST 训练集所有像素的均值和标准差（业界通用值），用于标准化
mnistMean = 0.1307
mnistStd = 0.3081


# ================================ 数据 ================================
def loadMnist(train):
    """读取 MNIST，返回 (图片, 标签)。图片形状 (N, 1, 28, 28)，像素取值 [0, 1]"""
    # download=True：本地没有数据时自动下载，已有则直接读取，不会重复下载
    dataset = datasets.MNIST(root=dataRoot, train=train, download=True)
    # 直接取出整个数据集的张量：uint8 (N, 28, 28) → float (N, 1, 28, 28)，0~255 缩放到 0~1
    # 不用 transforms 逐张处理，是因为 MNIST 很小（6 万张只占约 190MB），一次性转成张量更快
    images = dataset.data.unsqueeze(1).float() / 255
    return images, dataset.targets


def randomAffine(images):
    """
    数据增强：对一个批次里的每张图片分别做随机旋转、缩放、平移，凭空"造"出更多样的手写数字，
    逼模型学习笔画本身，而不是死记像素位置。
    整批一起在 GPU 上计算，比 torchvision 的 RandomAffine 在 CPU 上逐张处理快十几倍。
    原理：给每张图生成一个 2x3 的仿射矩阵，affine_grid 根据矩阵算出每个输出像素应该去原图哪里取值，
    grid_sample 再按这些坐标插值取值。
    """
    imageCount = images.size(0)
    device = images.device
    # 每张图随机的旋转角度（弧度）、缩放倍数、平移量
    angles = (torch.rand(imageCount, device=device) * 2 - 1) * math.radians(maxRotateDegrees)
    scales = torch.empty(imageCount, device=device).uniform_(*scaleRange)
    # affine_grid 的坐标范围是 [-1, 1]，整幅图宽度为 2，所以平移 10% 对应 0.2
    shifts = (torch.rand(imageCount, 2, device=device) * 2 - 1) * maxTranslate * 2

    # 旋转 + 缩放矩阵。除以 scales 是因为 affine_grid 描述的是"从输出坐标到原图坐标"的映射，
    # 取样范围缩小为 1/s，图像就放大为 s 倍
    cosValues = torch.cos(angles) / scales
    sinValues = torch.sin(angles) / scales
    theta = torch.stack([
        torch.stack([cosValues, -sinValues, shifts[:, 0]], dim=1),
        torch.stack([sinValues, cosValues, shifts[:, 1]], dim=1),
    ], dim=1)  # 形状 (imageCount, 2, 3)

    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    # 超出原图范围的位置补 0，正好是 MNIST 的黑色背景（所以增强要在标准化之前做）
    return F.grid_sample(images, grid, align_corners=False)


# ================================ 网络结构 ================================
def fcBlock(inFeatures, outFeatures):
    """一个隐藏层：全连接 → 批归一化 → ReLU 激活 → Dropout"""
    return nn.Sequential(
        nn.Linear(inFeatures, outFeatures, bias=False),  # 后面紧跟 BatchNorm，它自带偏移项，所以这里省掉 bias
        nn.BatchNorm1d(outFeatures),  # 批归一化：稳定每层输入的分布，训练更快更稳，也允许更大的学习率
        nn.ReLU(inplace=True),        # ReLU 激活：引入非线性，没有它多层全连接等价于一层
        nn.Dropout(dropoutRate),      # Dropout：随机丢弃神经元，防止过拟合
    )


class Net(nn.Module):
    """全连接网络：784 → hiddenSizes[0] → hiddenSizes[1] → ... → 10"""

    def __init__(self):
        super().__init__()
        layers = [nn.Flatten()]  # (batch, 1, 28, 28) → (batch, 784)，全连接层要求输入是二维的
        inFeatures = 28 * 28
        for outFeatures in hiddenSizes:  # 按 hiddenSizes 依次堆叠隐藏层
            layers.append(fcBlock(inFeatures, outFeatures))
            inFeatures = outFeatures
        # 输出层：10 个类别的得分（logits）
        # 注意：这里不加 Softmax！CrossEntropyLoss 内部已经包含了 log_softmax
        layers.append(nn.Linear(inFeatures, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # 标准化：(像素 - 均值) / 标准差，让输入大致是均值 0、方差 1，训练更稳定
        # 放在模型里面，训练、测试、以后拿模型做预测时都不会漏掉这一步
        x = (x - mnistMean) / mnistStd
        return self.network(x)


# ================================ 训练与评估 ================================
def trainOneEpoch(model, trainLoader, lossFunction, optimizer, scheduler, device):
    """训练一个 epoch，返回平均 loss 和训练集准确率"""
    model.train()  # 训练模式：启用 Dropout，BatchNorm 使用当前批次的统计量
    totalLoss, correctCount, sampleCount = 0.0, 0, 0
    for inputs, labels in trainLoader:
        # 把数据搬到模型所在的设备（GPU 或 CPU）
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        inputs = randomAffine(inputs)  # 数据增强只在训练时做，每个 epoch 看到的都是不同的变形

        outputs = model(inputs)               # 前向传播，得到形状 (batch, 10) 的得分
        loss = lossFunction(outputs, labels)  # 交叉熵直接使用整数标签，不需要 one-hot

        optimizer.zero_grad()  # 梯度清零（否则会和上一批的梯度累加）
        loss.backward()        # 反向传播，计算梯度
        optimizer.step()       # 根据梯度更新权重
        scheduler.step()       # OneCycle 策略每个 batch 调整一次学习率

        # 统计：loss.item() 是这一批的平均值，乘以批大小还原成总和
        batchCount = labels.size(0)
        totalLoss += loss.item() * batchCount
        correctCount += (outputs.argmax(dim=1) == labels).sum().item()
        sampleCount += batchCount
    return totalLoss / sampleCount, correctCount / sampleCount


@torch.no_grad()  # 评估时不需要计算梯度，省显存、更快
def evaluate(model, dataLoader, lossFunction, device):
    """在给定数据集上评估，返回平均 loss 和准确率"""
    model.eval()  # 评估模式：关闭 Dropout，BatchNorm 改用训练时累计的全局统计量
    totalLoss, correctCount, sampleCount = 0.0, 0, 0
    for inputs, labels in dataLoader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        batchCount = labels.size(0)
        totalLoss += lossFunction(outputs, labels).item() * batchCount
        correctCount += (outputs.argmax(dim=1) == labels).sum().item()  # 得分最高的类别即预测结果
        sampleCount += batchCount
    return totalLoss / sampleCount, correctCount / sampleCount


def main():
    torch.manual_seed(randomSeed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    useCuda = device.type == 'cuda'
    print(f'使用设备: {device}')

    # 装载数据集。pin_memory（锁页内存）只在用 GPU 时有意义，能加快 CPU → GPU 的数据拷贝
    trainImages, trainLabels = loadMnist(train=True)
    testImages, testLabels = loadMnist(train=False)
    trainLoader = DataLoader(TensorDataset(trainImages, trainLabels), batch_size=batchSize,
                             shuffle=True, pin_memory=useCuda)
    # 测试不需要反向传播、占用显存少，批次可以开大一些，评估更快
    testLoader = DataLoader(TensorDataset(testImages, testLabels), batch_size=1000,
                            shuffle=False, pin_memory=useCuda)

    model = Net().to(device)
    # 交叉熵损失 = log_softmax + 负对数似然，适合多分类
    # 用了标签平滑后 loss 不会降到接近 0，这是正常现象
    lossFunction = nn.CrossEntropyLoss(label_smoothing=labelSmoothing)
    # AdamW：自适应学习率的 Adam，并把权重衰减和梯度更新解耦，正则化效果更好
    optimizer = optim.AdamW(model.parameters(), lr=maxLearningRate, weight_decay=weightDecay)
    # OneCycle：学习率先从很小升到 maxLearningRate（热身），再慢慢降到接近 0（精细收敛）
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=maxLearningRate,
                                              epochs=epochNum, steps_per_epoch=len(trainLoader))

    for epoch in range(1, epochNum + 1):
        startTime = time.time()
        trainLoss, trainAcc = trainOneEpoch(model, trainLoader, lossFunction, optimizer, scheduler, device)
        testLoss, testAcc = evaluate(model, testLoader, lossFunction, device)
        # 训练准确率比测试准确率低是正常的：训练时有数据增强和 Dropout，任务更难
        print(f'epoch {epoch:2d}/{epochNum} | '
              f'训练 loss {trainLoss:.4f} 准确率 {trainAcc:.4f} | '
              f'测试 loss {testLoss:.4f} 准确率 {testAcc:.4f} | '
              f'耗时 {time.time() - startTime:.1f}s')

    # 保存最后一轮的模型。不按测试集挑"最好的一轮"：那相当于偷看了测试集，报告的准确率会偏高
    torch.save(model.state_dict(), modelPath)
    print(f'最终测试集准确率: {testAcc:.4f}，模型已保存到 {modelPath}')


if __name__ == '__main__':
    main()
