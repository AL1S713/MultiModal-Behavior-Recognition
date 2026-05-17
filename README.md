# MultiModal-Behavior-Recognition

基于RGB和红外（IR）双模态视频的行为识别框架。使用预训练SlowFast提取时空特征，结合可靠性评估与跨模态注意力融合动态融合双模态信息，实现高精度行为分类。

主要特性

- 双模态输入（RGB + IR），适应光照变化与夜间场景
- SlowFast架构，低帧率（slow）和高帧率（fast）捕获全局与局部运动
- 可靠性评估模块：基于特征方差与熵自动生成模态权重
- 跨模态注意力融合：根据可靠性权重增强重要模态贡献
- 混合精度训练（AMP），加速训练并节省显存
- Weights & Biases（WandB）自动记录训练曲线

环境配置

使用 Conda（推荐）

git clone https://github.com/your-username/MultiModal-Behavior-Recognition.git
cd MultiModal-Behavior-Recognition
conda env create -f environment.yml
conda activate MBR

核心依赖

- Python 3.10
- PyTorch 2.0.1+cu118
- torchvision 0.15.2+cu118
- pytorchvideo
- wandb
- numpy, tqdm, opencv-python

数据集准备

期望目录结构：

data/
├── rgb/          # RGB视频，命名: {prefix}_rgb.avi
└── ir/           # 红外视频，命名: {prefix}_ir.avi

- 视频文件通过前缀 {prefix} 成对匹配
- 自动取两个模态的最小时长，采样固定帧数
- 标签默认从文件名（如 A001_rgb.avi）中提取 A 后的数字

自定义数据集可继承 NTUDataset 并重写 _build_label_map。

模型架构

RGB → SlowFast(冻结) → 特征
IR  → SlowFast(训练) → 特征
          ↓                ↓
     可靠性评估模块 → 模态权重 (w_rgb, w_ir)
          ↓                ↓
     跨模态注意力融合 → 融合特征
                       ↓
                   分类头 → 类别logits

核心模块

模块: SlowFast特征提取器, 文件: slowfast_feature.py, 功能: 加载预训练SlowFast（去分类层），输出2304维特征
模块: 双模态提取器, 文件: multimodal_slowfast.py, 功能: 分别处理RGB和IR，IR单通道自动复制为3通道
模块: 可靠性评估, 文件: Fusion.py (Reliability), 功能: 计算方差和熵，MLP输出归一化权重
模块: 跨模态注意力, 文件: Fusion.py (CrossModalAttention), 功能: 权重加权注意力融合
模块: 分类头, 文件: ClassificationHead.py, 功能: 单层线性或两层MLP分类器

训练

1. 配置文件示例

创建 configs/default.yaml：

data:
  rgb_dir: "data/rgb"
  ir_dir: "data/ir"
  slow_num_frames: 8
  fast_num_frames: 32
  side_size: 256
  batch_size: 8
  num_workers: 4

model:
  rgb_weight: "pretrained/slowfast_rgb.pth"  # 可选
  ir_weight: null
  feature_dim: 2304
  num_classes: 10   # 修改为你的类别数
  hidden_dim: 1024
  dropout: 0.5

train:
  lr: 1e-4
  weight_decay: 1e-5
  epochs: 50
  warmup_epochs: 5
  grad_clip: 1.0
  use_amp: true
  save_dir: "checkpoints"
  eval_interval: 1

wandb:
  project: "MBR"
  entity: "your_entity"
  name: "exp1"

2. 启动训练

python train.py --config configs/default.yaml

- 最佳模型保存至 checkpoints/best_model.pth
- 训练指标自动上传至WandB（需先 wandb login）

验证与测试

加载最佳模型示例：

checkpoint = torch.load("checkpoints/best_model.pth")
# 重新构建模型并加载 state_dict

引用

@misc{MBR2026,
  author = {Your Name},
  title = {MultiModal Behavior Recognition with Reliability-Guided Cross-Modal Attention},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/your-username/MultiModal-Behavior-Recognition}
}

许可证

MIT License

致谢

PyTorchVideo: https://github.com/facebookresearch/pytorchvideo
SlowFast: https://github.com/facebookresearch/SlowFast

贡献

欢迎提交 Issue 和 PR。请遵循 PEP8 规范。
