
# 实验效果

本节系统评估了不同推理策略、不同 prompt、不同模型规模及 LoRA 微调对商品评论摘要任务的影响。
实验使用 **1280 条样本**，评测指标为 **ROUGE-1 F1** 与 **ROUGE-L F1**。

---

## 一、Zero-shot 推理

在不进行微调的情况下直接使用模型进行总结。
由于观察到**64 token 基本能覆盖摘要长度**，因此统一设置 `max_new_tokens=64`。


---

### 1. 默认参数

```bash
python3 inference_batch.py --model_name ./model_weights/Qwen/Qwen3-0.6B
```

**结果：**

Sampled 1280 entries -- ROUGE-1 F1: 0.0452, ROUGE-L F1: 0.0414

**示例：**

```json
{
  "review_body": "不好用，一股味道。包装不是密封，感觉被人用过？和日系比差远了。用了2次丢了。",
  "review_title": "不好用",
  "summary": "商品体验不佳，包装不密封、气味难闻，使用后丢弃，与日系产品相比存在明显差距。"
}
{
  "review_body": "正品，和德国直邮的袋装口感一样。宝宝爱喝",
  "review_title": "正品",
  "summary": "该商品是正品，并且与德国直邮的袋装口感一致，宝宝喜欢饮用。"
}
```

**分析：**
摘要风格较模板化、信息覆盖有限。

---

### 2. 采样生成（do_sample=True）

采样会提升生成的随机性，但摘要属于**高度确定性**任务，因此采样往往会拉低 ROUGE。

---

#### 1. do_sample=True（默认 temperature）

```bash
python3 inference_batch.py \
    --model_name ./model_weights/Qwen/Qwen3-0.6B \
    --min_new_tokens 1 \
    --do_sample True
```

**结果：**

Sampled 1280 entries -- ROUGE-1 F1: 0.0392, ROUGE-L F1: 0.0349

**示例：**

```json
{"review_body": "不好用，一股味道。包装不是密封，感觉被人用过？和日系比差远了。用了2次丢了。", "review_title": "不好用", "summary": "这款商品体验不佳，包装破损，气味不理想，使用后丢失，与日系产品相比有较大差距。"}
{"review_body": "正品，和德国直邮的袋装口感一样。宝宝爱喝", "review_title": "正品", "summary": "这款商品是正品，与德国直邮包装的袋装口感一致，宝宝喜欢。"}

```

**分析：**
信息偏移、摘要更冗余，导致 ROUGE 下降。

---

### 2. 极低 temperature（0.1）

```bash
python3 inference_batch.py \
    --model_name ./model_weights/Qwen/Qwen3-0.6B \
    --min_new_tokens 1 \
    --do_sample True \
    --temperature 0.1
```

**ROUGE：**

Sampled 1280 entries -- ROUGE-1 F1: 0.0449, ROUGE-L F1: 0.0407

**示例：**

```json
{"review_body": "不好用，一股味道。包装不是密封，感觉被人用过？和日系比差远了。用了2次丢了。", "review_title": "不好用", "summary": "这款商品体验不佳，味道不理想，包装未密封，使用后可能被他人使用过，与日系产品相比存在差距，且多次丢失。"}
{"review_body": "正品，和德国直邮的袋装口感一样。宝宝爱喝", "review_title": "正品", "summary": "该商品是正品，并且与德国直邮的袋装口感一致，宝宝喜欢饮用。"}
```


**分析：**
低温让采样近似 deterministic，但仍存在轻微复述偏差。

---

### 3. 超低 temperature（0.01）

```bash
--temperature 0.01
```

**ROUGE：**

Sampled 1280 entries -- ROUGE-1 F1: 0.0457, ROUGE-L F1: 0.0417

**示例：**

```json
{"review_body": "不好用，一股味道。包装不是密封，感觉被人用过？和日系比差远了。用了2次丢了。", "review_title": "不好用", "summary": "商品体验不佳，包装不密封，气味难闻，与日系产品相比有明显差距，使用后丢失。"}
{"review_body": "正品，和德国直邮的袋装口感一样。宝宝爱喝", "review_title": "正品", "summary": "该商品是正品，并且与德国直邮的袋装口感一致，宝宝喜欢饮用。"}
}
```

**分析：**
几乎等价于贪心搜索 → 采样作用弱化。

---

## 三、Prompt 优化

#### 原 Prompt（较弱）

> “你将看到一个来自电商平台的商品评论，请根据评论内容，生成简短总结。”

#### 新 Prompt（更明确、更结构化）

> “你是一个电商平台商品评论摘要生成器。你将根据用户的评论正文（review_body）生成一个简短的评论标题（review_title）。”

```bash
python3 inference_batch.py \
    --model_name ./model_weights/Qwen/Qwen3-0.6B \
    --min_new_tokens 1 \
    --output ./output/inference_result_prompt_plus.jsonl
```

**ROUGE：**

Sampled 1280 entries -- ROUGE-1 F1: 0.0565, ROUGE-L F1: 0.0534

**示例：**
```json
{"review_body": "不好用，一股味道。包装不是密封，感觉被人用过？和日系比差远了。用了2次丢了。", "review_title": "不好用", "summary": "商品体验不佳，包装不密封、气味难闻，使用后丢弃，与日系产品相比存在明显差距。"}
{"review_body": "正品，和德国直邮的袋装口感一样。宝宝爱喝", "review_title": "正品", "summary": "该商品是正品，并且与德国直邮的袋装口感一致，宝宝喜欢饮用。"}
```

**分析：**

* Prompt 明确任务约束
* 模型输出格式更接近目标摘要
* 提升幅度显著（约 +25%）


📌 **这一项比模型规模提升更有用**

---

## 四、不同模型规模的效果

### 1. Qwen3-1.7B

```bash
python3 inference_batch.py   --model_name ./model_weights/Qwen/Qwen3-1.7B --min_new_tokens 1 --output  ./output/inference_result_prompt_1.7B.jsonl --batch_size 16
```

**ROUGE：**

Sampled 1280 entries -- ROUGE-1 F1: 0.0479, ROUGE-L F1: 0.0444


**示例：**
```json
{"review_body": "不好用，一股味道。包装不是密封，感觉被人用过？和日系比差远了。用了2次丢了。", "review_title": "不好用", "summary": "产品使用体验差，包装不密封，易受潮，使用两次后丢失。"}
{"review_body": "正品，和德国直邮的袋装口感一样。宝宝爱喝", "review_title": "正品", "summary": "正品，口感与德国直邮袋装一致，宝宝喜爱。"}
```

**分析：**
稍好于 0.6B，但提升幅度不大。
原因：此任务主要受 prompt & alignment 影响，而非模型参数量。

---

### 2. Qwen3-8B

```bash
python3 inference_batch.py   --model_name ./model_weights/Qwen/Qwen3-8B --min_new_tokens 1 --output  ./output/inference_result_prompt_8B.jsonl --batch_size 16
```

**ROUGE：**

Sampled 1280 entries -- ROUGE-1 F1: 0.0469, ROUGE-L F1: 0.0432

**示例：**

```json
{"review_body": "不好用，一股味道。包装不是密封，感觉被人用过？和日系比差远了。用了2次丢了。", "review_title": "不好用", "summary": "该商品使用体验差，有异味，包装不密封且疑似被他人使用，质量逊于日系产品，仅使用两次即丢弃。"}
{"review_body": "正品，和德国直邮的袋装口感一样。宝宝爱喝", "review_title": "正品", "summary": "正品，口感与德国直邮一致，宝宝喜欢。"}
```


**分析：**

* 未微调的大模型在短摘要任务中不如小模型微调后强
* 大模型在 zero-shot 下可能“过度生成无关内容”

---

## 五、LoRA 微调（核心提升）

模型：**Qwen3-0.6B**
LoRA 微调后效果提升最显著。

---

### 1. checkpoint-10k

```bash
python3 inference_batch.py    --model_name ./model_weights/Qwen/Qwen3-0.6B --min_new_tokens 1 --lora_path ./output/checkpoint-10000 --output  ./output/inference_result_lora_10k.jsonl 
```

**ROUGE：**

Sampled 1280 entries -- ROUGE-1 F1: 0.0668, ROUGE-L F1: 0.0665

**示例：**

```json
{"review_body": "不好用，一股味道。包装不是密封，感觉被人用过？和日系比差远了。用了2次丢了。", "review_title": "不好用", "summary": "一般般"}
{"review_body": "正品，和德国直邮的袋装口感一样。宝宝爱喝", "review_title": "正品", "summary": "好"}
```

**示例：**
```json

```

**分析：**
模型开始学到“短摘要风格”，但表达过短。

---

### 2. checkpoint-20k

```bash
python3 inference_batch.py    --model_name ./model_weights/Qwen/Qwen3-0.6B --min_new_tokens 1 --lora_path ./output/checkpoint-20000 --output  ./output/inference_result_lora_20k.jsonl 
```

**ROUGE：**

Sampled 1280 entries -- ROUGE-1 F1: 0.0721, ROUGE-L F1: 0.0708

**示例：**

```json
{"review_body": "不好用，一股味道。包装不是密封，感觉被人用过？和日系比差远了。用了2次丢了。", "review_title": "不好用", "summary": "一般般"}
{"review_body": "正品，和德国直邮的袋装口感一样。宝宝爱喝", "review_title": "正品", "summary": "好用"}
```

**分析：**
20k 是最佳点，短、稳定、贴合任务目标（商品评论标题）。

---

### 3. checkpoint-30k

```bash
python3 inference_batch.py    --model_name ./model_weights/Qwen/Qwen3-0.6B --min_new_tokens 1 --lora_path ./output/checkpoint-30000 --output  ./output/inference_result_lora_30k.jsonl 

```

**ROUGE：**

Sampled 1280 entries -- ROUGE-1 F1: 0.0724, ROUGE-L F1: 0.0713

**示例：**

```json
{"review_body": "不好用，一股味道。包装不是密封，感觉被人用过？和日系比差远了。用了2次丢了。", "review_title": "不好用", "summary": "一般般"}
{"review_body": "正品，和德国直邮的袋装口感一样。宝宝爱喝", "review_title": "正品", "summary": "好"}
```

**分析：**

* 与 20k 接近
* 模型趋向固定风格，但保持高一致性
* 没有出现过拟合导致的退化

---

## 六、总览对比（完整 ROUGE-1 & ROUGE-L）

| 模型 / 策略                      | ROUGE-1 F1 | ROUGE-L F1 | 备注                     |
| ---------------------------- | ---------- | ---------- | ---------------------- |
| Zero-shot（默认）                | 0.0452     | 0.0414     | 有空输出                   |
| Zero-shot + min_new_tokens=1 | 0.0453     | 0.0414     | 摆脱空输出                  |
| do_sample=True               | 0.0392     | 0.0349     | 随机性降低准确率               |
| do_sample, temp=0.1          | 0.0449     | 0.0407     | 近似 deterministic       |
| do_sample, temp=0.01         | 0.0457     | 0.0417     | 几乎等同贪心                 |
| Prompt 优化                   | 0.0565     | 0.0534     | 提升显著                   |
| Qwen3-1.7B                   | 0.0479     | 0.0444     | 规模略好，但提升有限             |
| Qwen3-8B                     | 0.0469     | 0.0432     | Zero-shot 不如 prompt 优化 |
| LoRA 10k                     | 0.0668     | 0.0665     | 微调显著提升                 |
| LoRA 20k                     | 0.0721     | 0.0708     | 持续提升          |
| **LoRA 30k**                 | **0.0724** | **0.0713** | **最佳效果**            |

---

## 七、核心结论

#### 1. Prompt 远比模型规模更重要

在摘要任务中，prompt alignment 才是关键，而非参数量。

#### 2. LoRA 微调收益巨大

0.6B 微调后全面超越 8B zero-shot。

#### 3. 摘要任务不适合采样

采样模式会降低鲁棒性与精确对齐能力。
