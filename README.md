# full_connect_nn

**[中文](#中文) | [English](#english) | [日本語](#日本語)**

---

## 中文

用全连接神经网络（多层感知机，MLP）做 MNIST 手写数字识别（10 分类）。

### 当前版本

| 项目 | 设置 |
|---|---|
| 网络结构 | 784 → 1024 → 512 → 256 → 10；每个隐藏层是 Linear + BatchNorm + ReLU + Dropout(0.3) |
| 损失函数 | CrossEntropyLoss，标签平滑 0.1 |
| 优化器 | AdamW（权重衰减 1e-4）+ OneCycle 学习率（峰值 3e-3），训练 30 个 epoch |
| 数据增强 | 随机旋转 ±10°、平移 10%、缩放 0.9~1.1 倍，在 GPU 上整批完成 |
| **测试集准确率** | **99.38%**（1 万张测试图错 62 张） |
| 训练耗时 | RTX 2060 上约 2.5 分钟 |

所有超参数都在 `main.py` 开头，可以直接修改。

### 运行

```bash
pip install torch torchvision
python main.py
```

首次运行会自动下载 MNIST 到 `./data`，训练结束后模型保存为 `mnist_fc.pt`。
测试环境：Python 3.10、PyTorch 2.5.1、torchvision 0.20.1。

### GPU 温控

`gpuTempGuard.py` 会启动训练脚本，每 2 秒记录一次 GPU 的温度、利用率、功耗和显存；温度达到上限（默认 80°C）时自动终止训练。

```bash
python gpuTempGuard.py                       # 带温控运行 main.py
python gpuTempGuard.py --limit 75            # 把温度上限改成 75°C
python gpuTempGuard.py --gpu 0               # 只监控 0 号 GPU（多卡服务器上使用）
python gpuTempGuard.py -- other.py --foo 1   # 运行别的脚本
```

训练日志和 GPU 记录（CSV）保存在 `logs/` 目录。需要 NVIDIA 显卡和 `nvidia-smi`。

参考：在 RTX 2060 上训练 `main.py` 时，GPU 最高 45°C，平均利用率 31%，平均功耗 22 W。

### 调参记录

从训练集里划出 1 万张作为验证集（调参时不碰测试集），用剩下的 5 万张训练：

| 配置 | 验证集准确率 | 训练耗时（RTX 2060） |
|---|---|---|
| 不做数据增强 | 98.87% | 约 100 秒 |
| **当前默认配置** | **99.25%** | 约 100 秒 |
| 训练 60 个 epoch | 99.33% | 约 190 秒 |
| 隐藏层加宽到 2048-1024-512 | 99.32% | 约 105 秒 |
| Dropout 改为 0.2 | 99.24% | 约 95 秒 |
| 更强的数据增强（±15°、平移 12%、缩放 0.85~1.15） | 99.08% | 约 95 秒 |

0.1% 以内的差距（相当于验证集里的 10 张图）基本在随机波动范围内。

### 更新日志

见 [CHANGELOG.md](CHANGELOG.md)。最初的版本（MSELoss + one-hot，没有隐藏层）可以用 `git show 4742dbe:main.py` 查看。

---

## English

MNIST handwritten digit recognition (10 classes) with a fully connected neural network (multilayer perceptron, MLP).

### Current version

| Item | Setting |
|---|---|
| Architecture | 784 → 1024 → 512 → 256 → 10; each hidden layer is Linear + BatchNorm + ReLU + Dropout(0.3) |
| Loss | CrossEntropyLoss with label smoothing 0.1 |
| Optimizer | AdamW (weight decay 1e-4) + OneCycle learning rate (peak 3e-3), 30 epochs |
| Data augmentation | Random rotation ±10°, translation up to 10%, scaling 0.9–1.1, applied per batch on the GPU |
| **Test accuracy** | **99.38%** (62 errors out of 10,000 test images) |
| Training time | About 2.5 minutes on an RTX 2060 |

All hyperparameters are at the top of `main.py`.

### Usage

```bash
pip install torch torchvision
python main.py
```

The first run downloads MNIST to `./data`. The trained model is saved as `mnist_fc.pt`.
Tested with Python 3.10, PyTorch 2.5.1 and torchvision 0.20.1.

### GPU temperature guard

`gpuTempGuard.py` launches the training script, records GPU temperature, utilization, power draw and memory every 2 seconds, and stops training when the temperature reaches the limit (80°C by default).

```bash
python gpuTempGuard.py                       # run main.py with the guard
python gpuTempGuard.py --limit 75            # set the limit to 75°C
python gpuTempGuard.py --gpu 0               # monitor only GPU 0 (for multi-GPU servers)
python gpuTempGuard.py -- other.py --foo 1   # run a different script
```

Training logs and GPU records (CSV) are saved to `logs/`. Requires an NVIDIA GPU and `nvidia-smi`.

For reference: training `main.py` on an RTX 2060 peaked at 45°C, with 31% average utilization and 22 W average power draw.

### Tuning results

10,000 images were held out from the training set as a validation set (the test set was not used for tuning); the remaining 50,000 were used for training:

| Configuration | Validation accuracy | Training time (RTX 2060) |
|---|---|---|
| No data augmentation | 98.87% | ~100 s |
| **Current default** | **99.25%** | ~100 s |
| 60 epochs | 99.33% | ~190 s |
| Wider hidden layers 2048-1024-512 | 99.32% | ~105 s |
| Dropout 0.2 | 99.24% | ~95 s |
| Stronger augmentation (±15°, 12% shift, 0.85–1.15 scale) | 99.08% | ~95 s |

Differences under 0.1% (about 10 validation images) are within random variation.

### Changelog

See [CHANGELOG.md](CHANGELOG.md). The original version (MSELoss + one-hot, no hidden layer) can be viewed with `git show 4742dbe:main.py`.

---

## 日本語

全結合ニューラルネットワーク（多層パーセプトロン、MLP）による MNIST 手書き数字認識（10 クラス分類）です。

### 現在のバージョン

| 項目 | 設定 |
|---|---|
| ネットワーク構造 | 784 → 1024 → 512 → 256 → 10。各隠れ層は Linear + BatchNorm + ReLU + Dropout(0.3) |
| 損失関数 | CrossEntropyLoss（ラベルスムージング 0.1） |
| 最適化手法 | AdamW（重み減衰 1e-4）+ OneCycle 学習率（最大 3e-3）、30 エポック |
| データ拡張 | ランダム回転 ±10°、平行移動 最大 10%、拡大縮小 0.9〜1.1 倍（GPU 上でバッチ単位に処理） |
| **テスト精度** | **99.38%**（テスト画像 10,000 枚中 誤り 62 枚） |
| 学習時間 | RTX 2060 で約 2.5 分 |

ハイパーパラメータはすべて `main.py` の先頭で変更できます。

### 実行方法

```bash
pip install torch torchvision
python main.py
```

初回実行時に MNIST が `./data` に自動ダウンロードされます。学習済みモデルは `mnist_fc.pt` に保存されます。
動作確認環境：Python 3.10、PyTorch 2.5.1、torchvision 0.20.1。

### GPU 温度ガード

`gpuTempGuard.py` は学習スクリプトを起動し、GPU の温度・使用率・消費電力・メモリ使用量を 2 秒ごとに記録します。温度が上限（デフォルト 80°C）に達すると学習を自動停止します。

```bash
python gpuTempGuard.py                       # 温度ガード付きで main.py を実行
python gpuTempGuard.py --limit 75            # 上限を 75°C に変更
python gpuTempGuard.py --gpu 0               # GPU 0 のみ監視（マルチ GPU サーバー向け）
python gpuTempGuard.py -- other.py --foo 1   # 別のスクリプトを実行
```

学習ログと GPU の記録（CSV）は `logs/` に保存されます。NVIDIA GPU と `nvidia-smi` が必要です。

参考：RTX 2060 で `main.py` を学習した際は、最高 45°C、平均使用率 31%、平均消費電力 22 W でした。

### チューニング結果

学習データから 10,000 枚を検証データとして分け（チューニングにテストデータは使用していません）、残りの 50,000 枚で学習しました。

| 設定 | 検証精度 | 学習時間（RTX 2060） |
|---|---|---|
| データ拡張なし | 98.87% | 約 100 秒 |
| **現在のデフォルト** | **99.25%** | 約 100 秒 |
| 60 エポック | 99.33% | 約 190 秒 |
| 隠れ層を 2048-1024-512 に拡大 | 99.32% | 約 105 秒 |
| Dropout 0.2 | 99.24% | 約 95 秒 |
| 強めのデータ拡張（±15°、平行移動 12%、拡大縮小 0.85〜1.15） | 99.08% | 約 95 秒 |

0.1% 未満の差（検証画像 約 10 枚分）はランダムなばらつきの範囲内です。

### 更新履歴

[CHANGELOG.md](CHANGELOG.md) を参照してください。最初のバージョン（MSELoss + one-hot、隠れ層なし）は `git show 4742dbe:main.py` で確認できます。
