# 更新日志 / Changelog / 更新履歴

## v2.0 — 2026-11-14

### 中文

**测试集准确率达到 99.38%**（1 万张测试图错 62 张），RTX 2060 上训练约 2.5 分钟。

**模型与训练**
- 损失函数：MSELoss + Softmax + one-hot 标签 → CrossEntropyLoss（标签平滑 0.1）
- 网络结构：单层 784→10 → 三个隐藏层 784→1024→512→256→10，每层带 BatchNorm、ReLU、Dropout(0.3)
- 新增数据增强（随机旋转、平移、缩放），在 GPU 上整批完成，比在 CPU 上逐张处理快约 8 倍（每个 epoch 约 30 秒 → 约 4 秒）
- 输入标准化（均值 0.1307、标准差 0.3081），放在模型内部
- 优化器：SGD（lr=0.5）→ AdamW + OneCycle 学习率，训练 30 个 epoch

**问题修复**
- 删除未使用的 `from keras.backend import one_hot`（没装 Keras 时会直接报错），以及未使用的 `numpy`、`Variable` 导入
- `download=False` 在本地没有数据时会报错，改为 `download=True`
- 评估时使用 `torch.no_grad()` 和 `model.eval()`

**其他**
- 自动使用 GPU、固定随机种子、每个 epoch 打印 loss/准确率/耗时，训练结束后保存模型到 `mnist_fc.pt`
- 变量名统一改为驼峰命名，补充中文注释
- 新增 `gpuTempGuard.py`：训练时每 2 秒记录 GPU 温度，达到上限（默认 80°C）自动终止训练
- 新增 `.gitignore` 和本更新日志；README 改为中英日三语

### English

**Test accuracy: 99.38%** (62 errors out of 10,000 test images); training takes about 2.5 minutes on an RTX 2060.

**Model and training**
- Loss: MSELoss + Softmax + one-hot labels → CrossEntropyLoss (label smoothing 0.1)
- Architecture: a single 784→10 layer → three hidden layers 784→1024→512→256→10, each with BatchNorm, ReLU and Dropout(0.3)
- Added data augmentation (random rotation, translation and scaling) applied per batch on the GPU, about 8× faster than per-image augmentation on the CPU (≈30 s → ≈4 s per epoch)
- Input normalization (mean 0.1307, std 0.3081), done inside the model
- Optimizer: SGD (lr=0.5) → AdamW with a OneCycle learning-rate schedule, 30 epochs

**Fixes**
- Removed the unused `from keras.backend import one_hot`, which crashed when Keras is not installed, and the unused `numpy` and `Variable` imports
- `download=False` crashed when the dataset was not present locally; changed to `download=True`
- Evaluation now runs under `torch.no_grad()` with `model.eval()`

**Other**
- Uses the GPU automatically, fixes the random seed, prints loss/accuracy/time for every epoch, and saves the model to `mnist_fc.pt`
- Renamed variables to camelCase and added Chinese comments
- Added `gpuTempGuard.py`: records GPU temperature every 2 seconds during training and stops training when it reaches the limit (80°C by default)
- Added `.gitignore` and this changelog; the README is now in Chinese, English and Japanese

### 日本語

**テスト精度 99.38%**（テスト画像 10,000 枚中 誤り 62 枚）。RTX 2060 での学習時間は約 2.5 分です。

**モデルと学習**
- 損失関数：MSELoss + Softmax + one-hot ラベル → CrossEntropyLoss（ラベルスムージング 0.1）
- ネットワーク構造：1 層（784→10）→ 隠れ層 3 層（784→1024→512→256→10）。各層に BatchNorm・ReLU・Dropout(0.3) を追加
- データ拡張（ランダム回転・平行移動・拡大縮小）を追加。GPU 上でバッチ単位に処理するため、CPU で 1 枚ずつ処理する場合より約 8 倍高速（1 エポックあたり約 30 秒 → 約 4 秒）
- 入力の標準化（平均 0.1307、標準偏差 0.3081）をモデル内部で実施
- 最適化手法：SGD（lr=0.5）→ AdamW + OneCycle 学習率スケジュール、30 エポック

**不具合修正**
- 未使用の `from keras.backend import one_hot`（Keras 未インストール時にエラーになる）と、未使用の `numpy`・`Variable` のインポートを削除
- ローカルにデータがない場合に `download=False` でエラーになるため、`download=True` に変更
- 評価時に `torch.no_grad()` と `model.eval()` を使用

**その他**
- GPU の自動利用、乱数シードの固定、エポックごとの loss・精度・所要時間の表示、学習後のモデル保存（`mnist_fc.pt`）に対応
- 変数名をキャメルケースに統一し、中国語のコメントを追加
- `gpuTempGuard.py` を追加：学習中に GPU 温度を 2 秒ごとに記録し、上限（デフォルト 80°C）に達すると学習を自動停止
- `.gitignore` と本更新履歴を追加。README を中国語・英語・日本語の 3 言語対応に変更

## v1.0 — 2025-11-14

### 中文

初始版本：单层全连接网络 784→10，输出层 Softmax，MSELoss + one-hot 标签，SGD（lr=0.5），批大小 64，训练 10 个 epoch。

### English

Initial version: a single fully connected layer 784→10 with a Softmax output, MSELoss with one-hot labels, SGD (lr=0.5), batch size 64, 10 epochs.

### 日本語

初版：全結合層 1 層（784→10）、出力層に Softmax、MSELoss + one-hot ラベル、SGD（lr=0.5）、バッチサイズ 64、10 エポック。
