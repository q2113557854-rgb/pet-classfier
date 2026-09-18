# pet-project · 基于深度学习的牛津宠物细粒度图像分类（37 类）

科研项目组本科生考核 —— 3 天极速挑战 Day3 交付仓库。

使用 PyTorch 在 **Oxford-IIIT Pet**（37 个猫狗品种，约 7,349 张）上完成细粒度分类：
预训练 **ResNet-18** 微调 + **2 组单变量消融实验**（Label Smoothing、Freeze Backbone）+ 混淆矩阵 / Grad-CAM 可视化。

---

## 1. 目录结构

```text
pet_project/
├── data/
│   └── dataset.py       # 数据加载、70/15/15 分层划分与 Transform
├── models/
│   └── model.py         # ResNet-18 定义与分类头替换（支持冻结骨干）
├── utils/
│   ├── common.py        # 随机种子 / 设备 / 中文字体（小补充模块）
│   ├── metrics.py       # Top-1/Top-5、Macro-F1、混淆矩阵、训练曲线绘制
│   └── gradcam.py       # Grad-CAM 可视化工具
├── train.py             # 训练主入口（argparse 参数控制）
├── evaluate.py          # 评估与可视化独立脚本
├── requirements.txt     # 依赖清单
└── README.md
```

> `data/ models/ utils/` 为 Python 包（含 `__init__.py`）；`datasets/`、`checkpoints/`、`runs/`、
> `outputs/` 为运行产物（已被 `.gitignore` 忽略）。

## 2. 快速开始（Google Colab 一键复现）

在全新 Colab（T4 GPU）或本地环境中依次执行：

```bash
# ① 拉取代码
git clone https://github.com/q2113557854-rgb/pet-classfier.git
cd pet-project

# ② 安装依赖（Colab 已预装 torch，其余按需安装）
pip install -r requirements.txt

# ③ 跑通 Baseline（自动下载数据集到 ./datasets，约 800MB）
python train.py --epochs 10 --run_name baseline

# ④ 消融 1：Label Smoothing 0.1
python train.py --epochs 10 --label_smoothing 0.1 --run_name label_smooth0.1

# ⑤ 消融 2：冻结骨干网络，仅训练 FC 层
python train.py --epochs 10 --freeze_backbone --run_name freeze_backbone

# ⑥ 评估与可视化（混淆矩阵 + Grad-CAM + 指标）
python evaluate.py --checkpoint checkpoints/baseline/best_model.pth \
                   --curves_csv outputs/logs/baseline.csv

# ⑦ 查看训练曲线
tensorboard --logdir runs
```

预期：单组 10 轮训练在 T4 / RTX 4060 上约 **3~5 分钟**，测试集 Top-1 ≈ **90%**。

## 3. 实验协议（与考核指南一致）

| 项目 | 配置 |
| :--- | :--- |
| 数据集 | Oxford-IIIT Pet，37 类，共 7,349 张（trainval 3,680 + test 3,669） |
| 数据划分 | 70% 训练 / 15% 验证 / 15% 测试，**分层抽样**（StratifiedShuffleSplit, seed=42）<br>可选 `--split_mode official_test` 使用官方 test 集（3,669 张） |
| 训练增强 | Resize(256) → RandomCrop(224) → RandomHorizontalFlip → Normalize |
| 验证/测试增强 | Resize(256) → CenterCrop(224) → Normalize（**无随机增强，防泄漏**） |
| 模型 | `torchvision` 预训练 ResNet-18，替换 fc 层输出 37 |
| 优化器 | AdamW(lr=1e-4, weight_decay=0.01)，Batch Size=32 |
| 损失 | `nn.CrossEntropyLoss()`（自带 Softmax，输出层**不**再加 Softmax） |
| 种子 | seed=42（torch / numpy / random / cudnn 全部固定） |

自查要点：验证/测试集无随机增强 ✅；评估用 `model.eval()` + `with torch.no_grad():` ✅；
`optimizer.zero_grad()` 在 `loss.backward()` 前 ✅；输出层无重复 Softmax ✅。

## 4. 实验结果

### 4.1 Day2 消融实验（每项 10 轮，验证集选最优模型后评估测试集）

| 实验配置 | Top-1 Acc (%) | Top-5 Acc (%) | Macro-F1 | 最优验证 Acc (%) | 训练轮数 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1. Baseline（全量微调 + 基础增强） | **90.58** | 待补算 | **0.9057** | 93.66 | 10 ep |
| 2. + Label Smoothing 0.1 | 89.67 | 待补算 | 0.8958 | 92.21 | 10 ep |
| 3. + Freeze Backbone（只训 FC） | 84.06 | 待补算 | 0.8400 | 85.33 | 10 ep |

> Top-5 待补算：对已保存的 `best_model.pth` 运行 `evaluate.py` 即可得到
> （Colab 一键补算单元格见《提交清单与答辩自查.md》第 2 步）。
> Day2 三组实验的逐轮 Loss/Acc 收敛曲线见报告图 1（数据来自 Colab 训练日志）。

**结论**：
- **Label Smoothing 0.1 在本任务上未带来增益**（-0.91% Top-1）。数据集类别间外观差异明显且样本充足，
  过拟合不严重，平滑化反而压低了模型对正确类别的置信度；
- **Freeze Backbone 显著掉点**（-6.52% Top-1）。ImageNet 预训练特征与宠物细粒度品种判别不完全匹配，
  必须微调高层特征才能学到品种级差异；
- 微调策略的收益（方案 2）远大于损失函数微调（方案 1），符合细粒度分类常见经验。

### 4.2 Day1 基线运行记录（现象复现：最佳轮次不固定）

| 运行 | 划分方式 | 最优验证 Acc | 最优轮次 | 测试 Top-1 |
| :--- | :--- | :--- | :--- | :--- |
| Day1-15 轮 | trainval 内 70/15/15（2576/552/552） | 93.30 | 第 11 轮 | 89.31 |
| Day1-10 轮 | trainval 85/15 + 官方 test（3128/552/3669） | 92.93 | 第 6 轮 | — |

两次运行"最优轮次不同"（第 6 轮 vs 第 11 轮）的原因：**数据划分版本不同 + 训练随机性**。
这提醒我们：比较不同运行/方法时必须**固定数据划分与随机种子**，否则轮次层面的对比没有意义
（详见报告第 2 节）。

## 5. 复现与验证状态

- [x] 代码模块化重构完成，全部模块通过冒烟测试（导入 / 前向 / 冻结骨干 / Grad-CAM / 中文字体）；
- [x] 一键复现命令整理完毕（见第 2 节），依赖清单 `requirements.txt` 可直接安装；
- [ ] 本地全量训练**未执行**：本机网络无法下载约 800MB 数据集与 CUDA 版 PyTorch（已尝试
      pytorch.org / 清华 / 阿里云 / 上交等多镜像均失败），故实验数据全部来自 Google Colab
      的真实训练日志（见第 4 节，非编造）；
- [ ] 待办：在全新 Colab 会话中按第 2 节命令复现验证（单组 10 轮约 3~5 分钟）。

## 6. 常见问题（答辩 FAQ）

1. **输入 `[32,3,224,224]` 的 Batch 经过 ResNet-18 最后一层卷积（AvgPool 前）输出形状？**
   → `[32, 512, 7, 7]`（ResNet-18 经过 5 次下采样，224→7，输出通道 512）。
2. **为什么验证/测试集不能用 RandomCrop / RandomFlip？**
   → 保持测试环境的确定性与客观性：结果可复现、不受随机干扰，才能公平比较不同模型。
3. **训练 Loss 很低但验证 Loss 飙升说明什么？如何缓解？**
   → 过拟合。缓解手段：数据增强（本项目）、Weight Decay、Early Stopping（本项目按验证集保存最优模型）。
4. **为什么 `optimizer.zero_grad()` 必须放在每个 Batch 训练循环开头？**
   → PyTorch 默认累积梯度，不清零会把上一个 batch 的梯度叠加进来，导致更新方向错误。

## 7. 已知问题与 AI 辅助说明

- **matplotlib 中文字体**：Colab 默认 DejaVu Sans 无中文字形，图标题会显示为 `□□`。
  本项目 `utils/common.py` 提供 `setup_cjk_font()`，找不到中文字体时自动改用英文标签规避（见报告第 4 节 Bug 复盘）。
- 本项目在数据加载、训练循环、Grad-CAM 等环节使用 AI 辅助生成代码，并人工逐行核查
  Tensor 维度、梯度清零、模式切换等易错点。
