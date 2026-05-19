---
title: "TTS模型微调：用自己的声音训练语音模型"
slug: tts-finetuning
description: "TTS 模型微调实战:用 XTTS、Fish Speech 训练你自己的声音,语音克隆的完整步骤。"
date: 2026-01-16T10:00:00+08:00
draft: false
tags: ["TTS", "XTTS", "Fish Speech", "模型微调", "语音克隆"]
categories: ["语音技术"]
excerpt: "有了数据，怎么训练自己的TTS模型？这篇用XTTS和Fish Speech两个开源模型举例，讲清楚微调的完整流程。"
---

## 两个主流开源方案

| 模型 | 特点 | 数据需求 | 显存要求 |
|------|------|----------|----------|
| XTTS v2 | 多语言，效果稳定 | 2-20分钟 | 12GB+ |
| Fish Speech | 中文效果好，速度快 | 3-10秒起 | 4GB+ |

---

## 方案一：XTTS微调

### 准备工作

**硬件要求**：
- GPU：12GB显存以上（推荐16GB）
- 内存：16GB以上

**数据要求**：
- 至少2-3分钟清晰录音
- 推荐5-20分钟效果更好
- WAV格式，16kHz以上

### 安装

```bash
git clone https://github.com/daswer123/xtts-finetune-webui
cd xtts-finetune-webui
pip install -r requirements.txt
```

### 数据格式

```
dataset/
├── audio/
│   ├── 001.wav
│   ├── 002.wav
│   └── ...
└── metadata.csv
```

metadata.csv 格式：
```
audio_file|text|speaker_name
audio/001.wav|今天天气真不错。|my_voice
audio/002.wav|我们去公园散步吧。|my_voice
```

### 训练配置

```yaml
# 关键参数
batch_size: 2          # 显存不够就调小
epochs: 10-50          # 数据少就多跑几轮
learning_rate: 5e-6    # 别调太大，容易过拟合
```

### 常见问题

**问题1：训练后声音变奇怪**
→ 过拟合了，减少epochs或增加数据

**问题2：声音不像**
→ 数据太少或质量不好，检查录音

**问题3：显存不够**
→ 减小batch_size，或用gradient accumulation

---

## 方案二：Fish Speech微调

Fish Speech对中文友好，而且显存要求低。

### 安装

```bash
git clone https://github.com/fishaudio/fish-speech
cd fish-speech
pip install -e .
```

### 零样本克隆（不用训练）

Fish Speech支持用3-10秒音频直接克隆：

```python
from fish_speech import FishSpeech

model = FishSpeech()
# 用参考音频生成
audio = model.generate(
    text="这是克隆后的声音",
    reference_audio="reference.wav"
)
```

### 微调（效果更好）

如果想要更像的效果，可以微调：

```bash
# 准备数据
python tools/prepare_data.py --input-dir ./my_audio --output-dir ./dataset

# 开始微调
python train.py --config configs/finetune.yaml --data-dir ./dataset
```

### 推理

```python
# 使用微调后的模型
audio = model.generate(
    text="现在声音更像了",
    voice_id="my_custom_voice"
)
```

---

## 对比选择

| 场景 | 推荐 |
|------|------|
| 快速验证 | Fish Speech（零样本） |
| 中文场景 | Fish Speech |
| 多语言 | XTTS |
| 最高质量 | XTTS微调20分钟数据 |

---

## 训练技巧

### 1. 不要贪多
- 10分钟高质量数据 > 1小时有底噪数据

### 2. 监控过拟合
- 训练loss下降但生成效果变差 → 停止训练

### 3. 多做对比
- 保存多个checkpoint，对比选最好的

### 4. 参考音频很重要
- XTTS生成时用的参考音频影响很大
- 选一段最清晰、最有代表性的

---

## 部署

训练好的模型可以用API服务起来：

```python
from fastapi import FastAPI
from fish_speech import FishSpeech

app = FastAPI()
model = FishSpeech(checkpoint="my_model")

@app.post("/tts")
def generate_speech(text: str):
    audio = model.generate(text)
    return {"audio": audio}
```

---

有问题留言。
