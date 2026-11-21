# 机器人表情响应预测

给定当前说话者的文本、声音和画面，以及机器人的听者角色，预测**机器人应做出的表情和动作强度**。目标不是识别说话者的情绪。训练和推理均可直接使用已有特征缓存，无需重新运行编码器。数据内容、标注与使用限制见 [数据集文档](dataset/DATASET.md)。

## 快速开始

以下命令均在项目根目录执行。使用 `uv sync --locked` 按 `uv.lock` 准备 Python 环境；已有环境可直接使用下面的 `uv run` 命令。

**评估已有模型，不训练：**

```bash
uv run --frozen python -m train.evaluator experiments/audio_compression/B3/seed_42/best.pt
```

**对一个已缓存的话轮推理：**

```bash
uv run --frozen python -m train.predict \
  experiments/audio_compression/B3/seed_42/best.pt T_000038 R01 --split train
```

预测入口接受 `train` 或 `val` 的话轮 ID 和听者角色，输出 JSON，包含表情、最终强度以及两个预测头的概率。它不是直接接受任意音视频文件的接口。

**开始一次新训练：**

```bash
uv run --frozen python -m train.main audio_compression \
  --run b3_reproduction --experiments B3 --seeds 42 --device auto
```

`--run` 是研究目录下的新目录名，已存在时会拒绝覆盖。上例只训练 B3 的一个种子；省略 `--experiments` 和 `--seeds` 时，会执行该研究配置列出的全部模型和种子。`--device auto` 在 CUDA 可用时使用 GPU，否则使用 CPU。

## 技术栈与代码职责

训练基于 **Python 和 PyTorch**。MiniLM、WavLM 和 MediaPipe 分别编码文本、音频和视频；NumPy 读取特征缓存，PyArrow 读写 Parquet，SwanLab 记录实验。

| 路径 | 职责 |
|---|---|
| `models/` | 本地预训练编码器及 MediaPipe 模型文件，不存放融合模型的训练结果 |
| `train/encoders.py`、`build_cache.py` | 提取并缓存三种模态的特征，与融合模型训练分开 |
| `train/data.py`、`features.py` | 对齐话轮、角色与缓存，构建 DataLoader；视频归一化参数只从 Train 拟合 |
| `train/architectures/` | `ModelConfig` 与 `build_model`：选择模态、音频投影和角色条件，构建网络 |
| `train/main.py`、`trainer.py`、`losses.py` | 读取实验配置，管理多种子训练、损失、早停和最佳权重 |
| `train/evaluator.py`、`metrics.py`、`predict.py` | 验证集评估、跨种子汇总、配对 Bootstrap 和单话轮推理 |
| `train/config.py`、`tracking.py` | 训练参数、历史 Checkpoint 兼容加载和 SwanLab 追踪 |
| `experiments/` | 按研究问题保存配置与结果；`feature_diagnostics/` 保留特征诊断记录 |

## 模型怎样生成预测

模型只读取启用的特征和角色。真实标签、质量标记及 `turn_id` 留给训练和评估，不作为预测证据。下表中 `B` 表示一个批次展开后的角色记录数，而不是话轮数。

| 输入 | 形状 | 处理方式 |
|---|---|---|
| `text` | `[B, 384]` | 投影 MiniLM 文本向量 |
| `audio` | `[B, 13, 3, 768]` | 学习 13 个 WavLM 层的权重，加权汇合各层，再投影三组池化特征 |
| `video` | `[B, 162]` | 投影由面部运动与头姿统计得到的话轮向量 |
| `role` | `[B]` | 通过 FiLM 调制融合向量，或与融合向量拼接；无角色对照组跳过此步 |

各启用模态默认投影到 128 维，拼接后经过融合层，再引入角色条件。两个输出头分别产生 `expression_logits [B, 8]` 和 `intensity_logits [B, 2]`。强度头区分 `low` 与 `medium+`；训练时只对真实非 E00 样本计算强度损失，推理时若预测为 E00，则最终强度固定为 `low`。

## 实验配置与比较

`experiments/<研究问题>/recipe.json` 定义模型、随机种子、早停策略和要比较的模型对。切换现有结构只需改配置，不需要复制 Trainer。也可用 `--config <路径>` 指定另一份配置。

| 研究配置 | 要比较的问题 | 如何选最佳 Checkpoint |
|---|---|---|
| [baseline](experiments/baseline/recipe.json) | `M0_CE` 与 `M0` 比较表情损失是否加权；`M0`–`M3` 比较模态组合；`M3_concat`、`M3_no_role` 与 `M3` 比较角色条件 | 最大化 Val Strict 八类表情 Macro-F1；最多 80 个 epoch，耐心值 15 |
| [audio_compression](experiments/audio_compression/recipe.json) | `B0` 仅文本，`B1` 加入音频，`B2` 再加入视频；`B3` 保留三模态，仅替换音频投影 | 最小化 Val Strict 加权损失的三轮滑动平均；最多 50 个 epoch，耐心值 8 |

比较音频压缩本身时应使用 **B2 对 B3**：B2 直接将 2,304 维音频向量投影到 128 维；B3 对三组 768 维向量共享一个 `768 → 64` 投影，拼接后再从 192 维投影到 128 维。两者都用 Concat 引入角色。

训练按话轮打乱并成批，再展开准入角色记录；优化器为 AdamW，配合余弦退火和梯度裁剪。训练结束后恢复最佳权重，分别报告 Val Strict 与 Val All 指标，包括表情 Macro-F1、逐类分数、非 E00 强度 AUROC、表情与强度联合准确率，以及双角色分歧指标。模型间的配对 Bootstrap 以话轮为重采样单位，不拆开同一话轮的两个角色。

## 实验产物与 SwanLab

新训练写入 `experiments/<研究问题>/<run>/`：`config.json` 保存实际配置、归一化参数和溯源摘要，`summary.json` 汇总种子与模型对比。每个 `<模型>/seed_<种子>/` 下有 `best.pt`、`history.csv`、`metrics.json` 和 `val_predictions.parquet`。加载历史权重时保留其相对目录结构：加载器需要对应的 `config.json` 重建网络并取得归一化参数。

每个模型和种子对应一个 SwanLab Run，项目名为 `robot-expression`。`--swanlab-mode offline` 是默认值，曲线和最终指标记录在种子目录的 `swanlog/`；`disabled` 关闭追踪但仍保存本地训练结果；`online` 用于已自行登录后的在线同步。默认不向 SwanLab 发送原始媒体、转录文本或模型权重。仅在显式添加 `--model-artifact` 时，将本次最佳权重及 `inference_config.json` 附到对应 Run。
