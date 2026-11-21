# Expression v1.1 · 数据集使用说明

## 1. 数据集概览

这是面向双角色条件仿生机器人面部表情响应生成的多模态单轮对话数据集：给定当前说话者的音视频、官方文本与听者角色，预测**听者应做出的表情类别和动作强度**，而不是识别说话者的情绪。

| 指标 | 冻结交付规模 |
|---|---|
| 独立话轮（`turn_id`） | 1,358；其中 `tier_1_strict` 902，`tier_2_relaxed` 456 |
| 角色标注 | 2,716 条；每轮各有一条 R01 和 R02 标注，共用该轮媒体 |
| 独立 Speaker / Session / Prompt | 19 / 14 / 50；按输入清单的 `speaker_id`、`session_id`、`prompt_id_unique` 分别去重 |
| 媒体资产 | `dataset/turns/` 下 1,358 个话轮目录、5,432 个非空文件；总计 4,215,690,247 字节（十进制 4.215690247 GB） |
| 版本组合 | `phase1_frozen + expression_v1.1 + scheme1_v1` |

训练、验证均从 v1.1 正式表加载。两表中的 `tier` 是**话轮**分层；`high_conf_train_eligible` 是**角色记录**级准入标记，二者不可互相替代。

## 2. 目录结构与核心文件索引

路径均相对于项目根目录。下表中 `P/` 指 `dataset/metadata/phase2_production/`；同名 Parquet、CSV 是同一份标注的两种格式，**不要拼接**。

| 路径 | 格式与用途 |
|---|---|
| `dataset/metadata/dataset_catalog.json` | JSON；权威入口，含相对路径、文件摘要、Schema、划分与加载口径 |
| `dataset/turns/<turn_id>/` | 每轮一组媒体与元数据，组成见下表 |
| `dataset/metadata/phase1_model_inputs_manifest.jsonl` | JSONL；每轮一行，从 `model_inputs` 获取允许使用的媒体路径与文本 |
| `dataset/metadata/phase2_splits/scheme1_v1/` | `turns.csv` 是固定成员表；`split_manifest.json` 记录划分约束，`prompt_repeats.csv` 记录 Prompt 复用情况 |
| `P/train_1000_annotations.parquet`、`P/train_1000_annotations.csv` | Train 正式角色标注；推荐 Parquet 以保留列表与布尔类型 |
| `P/val_192_annotations.parquet`、`P/val_192_annotations.csv` | Val 正式角色标注；保留全量用于对照 |
| `P/test_166_annotations_sealed.parquet`、`P/test_166_annotations_sealed.csv` | Test 封存标注；仅限最终评估，摘要见第 4 节 |
| `P/stage_a_audio_notes_checkpoint.jsonl` | JSONL；按话轮关联的结构化听觉笔记 |
| `P/expression_v1_1_frozen_spec.json`、`P/train_val_production_report.json` | JSON；分别为冻结标签与证据规范、Train/Val 正式统计 |
| `dataset/metadata/phase2_pilot/` | v1.0 对照基线：`pilot80_annotations.parquet`、`pilot80_annotations.csv`、`pilot80_analysis_report.json`；不并入 v1.1 训练表 |

| 单轮文件 | 含义与用法 |
|---|---|
| `speaker_A.mp4` | 当前说话者画面，视觉输入 |
| `speaker_A.wav` | 同一话轮的说话者音频，听觉输入 |
| `listener_B_ref.mp4` | 听者参考画面，仅限离线参考，不是输入也不是监督标签 |
| `meta.json` | 话轮元数据与审核信息；不能整份传入模型 |

**音画同步：**按同一 `turn_id` 配对视频和独立 WAV。交付话轮的视频轨局部起点为 0；说话者视频轨与 WAV 的时长绝对差最大约 32.67 ms，不应假定两者时长逐样点相等。该值按总档案索引的 `dataset/metadata/phase1_final_candidates.parquet` 中 strict + relaxed 话轮核对。第 3 节的 16 kHz 是标注阶段的解码规格，不是对磁盘源文件采样率的描述。

## 3. 标签体系与双角色条件（Expression v1.1）

| `expression_id` | 中文名 | `expression_en` | 语义与分类边界 |
|---|---|---|---|
| E00 | 自然专注 | `attentive` | 自然关注、平静倾听；不是发呆或无法判断。默认 `low`。 |
| E01 | 友好微笑 | `friendly_smile` | 温和友善、赞同或积极回应；不要求明显的幽默或兴奋。 |
| E02 | 愉悦轻笑 | `amused` | 对幽默、趣事或明显愉悦作出回应；比 E01 更有娱乐性。 |
| E03 | 关切共情 | `concerned` | 面对困难、担忧或脆弱情绪表达理解与关心；不直接模仿悲伤。 |
| E04 | 惊讶 | `surprised` | 回应意外消息或突然转折；惊异不等于没听懂。 |
| E05 | 疑惑求证 | `confused` | 信息不确定、矛盾或难以理解，需要进一步确认；疑问句不必然需要求证。 |
| E06 | 认真思考 | `thoughtful` | 对需分析、理解或推理的信息表现认知投入；认真思考不等于疑惑。 |
| E07 | 严肃克制 | `serious` | 面对严肃议题、冲突或边界问题，保持认真、克制；不是愤怒。 |

| `intensity` | 听者的面部动作幅度（不是说话者的音量或激动程度） |
|---|---|
| `low` | 轻微、克制，但能自然识别 |
| `medium` | 清晰、自然，适合常规面对面交互 |
| `high` | 明显高幅度，具有鲜明视觉表现的情绪峰值动作 |

`status` 与表情类别独立：`labeled` 表示已作出类别判断；仅当缺失信息、歧义或证据不可靠且足以改变回应时使用 `uncertain`。不因 relaxed 分层自动弃权；`uncertain` 不占用 E00，也不作为高置信训练标签。

| `role_id` | 行为风格（不含性别标签） | 表达倾向 |
|---|---|---|
| R01 | 温柔沉稳型 / Gentle & Composed | 以柔和动作承接倾诉、温和回应喜悦；通常为 low 至 medium，克制不等于冷漠。 |
| R02 | 粗犷外放型 / Bold & Expressive | 明确的情绪触发可有更鲜明的回应；中性陈述仍应克制，不因角色外放就提高强度。 |

**标注证据链 `cascaded_audio1.5_plus_sol`：**Stage A（`gpt-audio-1.5`）把当前 WAV 解码为 16 kHz 单声道 PCM，盲听后只生成客观听觉笔记，不选表情。Stage B（`gpt-6.2-sol`）结合官方文本、说话者视频在 15% / 40% / 65% / 90% 处的四帧与听觉笔记，独立标注两个角色，不接收原始 WAV、`rough_heard_words`、`listener_B_ref.mp4` 或另一角色的输出。

## 4. 防泄漏划分与 Train / Val 分布（scheme1_v1）

以冻结成员表为准：三个集合的 Speaker 两两互斥，同一 `turn_id` 的 R01/R02 始终同集，Train 与 Val、Train 与 Test 均无 Session 重叠。这不意味着 Val 与 Test 的 Session 也互斥，更不意味着按 Prompt 留出。

| 集合 | 话轮 | 角色记录 | strict / relaxed（话轮） | 使用口径 |
|---|---:|---:|---|---|
| Train | 1,000 | 2,000 | 695 / 305 | 高置信训练准入 1,941 条角色记录 |
| Val | 192 | 384 | 99 / 93 | 主评测用 strict（198 条角色记录）；另报全量对照 |
| Test | 166 | 332 | — | 盲态封存，仅作最终评估 |

**Test 文件 SHA-256**（对文件字节；路径前缀同第 2 节）：

| 封存文件 | SHA-256 |
|---|---|
| `P/test_166_annotations_sealed.parquet` | `375ea91688bf2d6a81cd73eb366a6b89b07e02f4aa0d79af9393546ad52d4ff2` |
| `P/test_166_annotations_sealed.csv` | `e5a0376812bc129d6640126d00ed26dbb7fe1cae20c38277dae65631dcfbf4fb` |

**表情 × 强度**（全量角色记录，R01 与 R02 合计，未做准入过滤）：

| 类别 | Train low | Train medium | Train high | Val low | Val medium | Val high |
|---|---:|---:|---:|---:|---:|---:|
| E00 | 638 | 0 | 0 | 151 | 0 | 0 |
| E01 | 507 | 140 | 0 | 72 | 25 | 0 |
| E02 | 32 | 83 | 2 | 7 | 13 | 0 |
| E03 | 134 | 126 | 0 | 25 | 15 | 0 |
| E04 | 27 | 35 | 0 | 8 | 9 | 0 |
| E05 | 30 | 14 | 0 | 5 | 1 | 0 |
| E06 | 170 | 12 | 0 | 40 | 1 | 0 |
| E07 | 25 | 25 | 0 | 4 | 8 | 0 |

| 配对或质量指标 | Train | Val |
|---|---:|---:|
| 双角色类别一致（以话轮为分母） | 819 / 1,000（81.9%） | 162 / 192（84.375%） |
| 仅同类别话轮：强度 R02 > R01 / 相等 / R02 < R01 | 300 / 519 / 0 | 52 / 110 / 0 |
| `high_conf_train_eligible` 为真 / 全部角色记录 | 1,941 / 2,000 | 376 / 384 |
| 未准入（去重角色记录数） | 59 | 8 |
| `audio_not_audible` | 54 | 8 |
| `status_uncertain`（`status == 'uncertain'`） | 3 | 0 |
| `low_confidence` | 1 | 0 |
| `manual_context_audit_excluded` | 2 | 0 |

质量标记可重叠，不能把各标记次数直接相加作为剔除数。Train 的 `T_002198` 缺少指代的比较对象，单轮无法确定唯一监督真值：两个角色的原始标签均保留，但都不准入。Val 的准入标记仅描述质量，不是主评测的过滤条件，也不授权用于训练。

## 5. 数据字段说明（Schema）

以下为加载所需字段；类型采用总档案中的 Arrow 表记。**主标注表每行是一个角色，另两表每行是一个话轮**，因此从标注表连接后两表时用 `turn_id` 作多对一连接。

| 表 | 关键字段与类型 | 用途 |
|---|---|---|
| 主标注 Parquet | `turn_id`, `role_id`: `large_string` | 联合主键；角色是条件，不是另一段媒体 |
| 同上 | `split`, `tier`, `speaker_id`, `session_id`, `conversation_id`, `transcript`: `large_string` | 集合、分层、来源与官方文本 |
| 同上 | `expression_id`, `expression_en`, `intensity`, `status`, `confidence`, `reason`, `alternative_expression_id`: `large_string` | 目标类别、强度及判断依据；备选类别可为空 |
| 同上 | `high_conf_train_eligible`: `bool`；`quality_flags`, `observed_modalities`, `pragmatic_tags`: `list<element: string>` | 训练筛选、质量和模态记录；不作为模型证据 |
| 输入白名单 JSONL | `turn_id`, `candidate_tier`: `string`；`source_provenance`: `struct` | `turn_id` 为主键；来源对象含 `speaker_id`、`session_id`、`prompt_id_unique` 等字符串 |
| 同上 | `model_inputs`: `struct<speaker_video: string, speaker_audio: string, transcript: string>` | 仅从此对象取说话者媒体相对路径与官方文本 |
| 听觉笔记 JSONL | `turn_id`: `string`；`notes`: `struct` | `turn_id` 为主键；两个角色共享同一份 Stage A 笔记 |
| `notes` 子字段 | `vocal_tone`, `arousal_level`, `speech_pace_and_pauses`, `irony_or_sarcasm_cue`, `audio_summary_for_annotator`: `string`；`paralinguistic_events`: `list<item: string>` | 可供 Stage B 使用的语气、唤醒水平、节奏停顿、副语言事件及摘要 |
| `notes` 子字段 | `audio_audible`: `bool`；`rough_heard_words`: `string` | 可闻性检查与盲听辅助记录；均不在 Stage B 的笔记输入白名单内 |

冻结的标注规范、划分和生产审计记录按原字节保留。它们内部的 `metadata/` 路径以 `dataset/` 为基准；历史摘要针对迁移前的白名单。总档案同时记录了新旧白名单摘要，仅两项说话者媒体路径的前缀改变。

## 6. 快速加载指南

从项目根目录运行。示例从总档案解析路径，只打开 Train、Val 和输入白名单；保留全量 Val 以便对照。`pandas` 使用项目环境中的 Parquet 读取引擎。

```python
import json
from pathlib import Path
import pandas as pd

root = Path.cwd()
catalog = json.loads((root / 'dataset/metadata/dataset_catalog.json').read_text(encoding='utf-8'))
rules = catalog['usage_rules_and_limitations']
train = pd.read_parquet(root / rules['training']['path'])
train = train.loc[train['high_conf_train_eligible']].copy()
val_all = pd.read_parquet(root / rules['validation']['path'])
val = val_all.loc[val_all['tier'].eq('tier_1_strict')].copy()

manifest = next(e['path'] for e in catalog['frozen_files_inventory']
                if Path(e['path']).name == 'phase1_model_inputs_manifest.jsonl')
wanted = set(train['turn_id']) | set(val_all['turn_id'])
with (root / manifest).open(encoding='utf-8') as f:
    inputs = pd.DataFrame([
        {'turn_id': r['turn_id'], **r['model_inputs']}
        for r in map(json.loads, f) if r['turn_id'] in wanted
    ])
for col in ('speaker_video', 'speaker_audio'):
    inputs[col] = inputs[col].map(lambda path: root / path)

def attach_inputs(labels):
    # 从白名单取 transcript，避免与标注表的同名列冲突。
    data = labels.drop(columns='transcript').merge(inputs, on='turn_id', how='left',
                                                   validate='many_to_one')
    assert data[['speaker_video', 'speaker_audio', 'transcript']].notna().all().all()
    return data

train, val = attach_inputs(train), attach_inputs(val)
# 模型证据：speaker_video、speaker_audio、transcript；条件：role_id。
# 监督目标：expression_id、intensity。不要把整行当作输入。
```

## 7. 使用规范与已知限制（必读）

- **输入与划分边界：**严禁输入 `listener_B_ref.mp4`、第一阶段 L3 审核字段、`prev_other_*` 或另一角色的输出。不要重做逐行随机划分，也不要将同一 `turn_id` 的两个角色拆到不同集合。
- **Test 纪律：**开发、调参、选择 Prompt 或制定质量规则时不得读取 `test_166_annotations_sealed.*`。仅在模型和实验设计锁定后，用封存集执行唯一一次最终评估。
- **强度监督稀缺：**全量 Train 只有 2 条 `high`，Val 为 0。原始标注保留 `low`、`medium`、`high` 三级；`high` 样本不足以支持对该级别的单独评估。
- **泛化边界：**`scheme1_v1` 的 Test 是当前分布下的跨说话者基准，不是全新 Prompt 基准。存在跨集 Prompt 复用与少量早期提示词诊断暴露；最终泛化能力需用独立外部数据盲测，交互得体性需辅以人类观察者评价。
