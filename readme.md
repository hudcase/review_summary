# 商品评论摘要生成流程

本项目基于电商平台商品评论，使用大语言模型（LLM）生成简短摘要，支持中、日、英三种语言。
.
├── dataset/                # 存放原始和划分后的训练/验证/测试数据集
├── dataset_process.py      # 数据格式转换、切分等预处理脚本
├── download_model.py       # 下载预训练模型或权重文件的脚本
├── fintune_llm.py          # 大语言模型微调主程序（训练入口）
├── inference.py            # 单样本推理脚本
├── inference_batch.py      # 批量推理脚本
├── loss_curve.png          # 训练过程中损失曲线可视化图像
├── model_weights/          # 存放训练后或下载得到的模型权重
├── output/                 # 推理结果、日志或中间输出文件目录
├── requirements.txt        # 项目依赖的 Python 包列表
├── readme.md               # 项目说明文档与使用指南
├── train_loss.txt          # 训练损失的文本记录文件
├── utils.py                # 公共工具函数（如日志、评估、数据处理等）
└── 结果分析.md             # 实验结果与性能分析文档（中文说明）

---

## 环境安装

请确保已安装 Python ≥ 3.10，并创建虚拟环境：
### 配置代理（国内网络访问不了huggingface）
source /etc/network_turbo

### 安装环境
```bash
pip install -r requirements.txt
```

> requirements.txt 中包含 `transformers`, `torch`, `peft`, `datasets`, `rouge_score` 等依赖。

---

## 1. 数据下载及处理

项目使用的原始评论数据集包含六种语言，我们仅使用其中的英语、中文、日语三种。

数据集来源：[Amazon Reviews Multi-Language](https://www.kaggle.com/datasets/mexwell/amazon-reviews-multi/data)

### 运行方式
运行以下命令处理数据：

```bash
python3 dataset_process.py
```

处理后，生成训练集、验证集和测试集：

| 数据集 | 样本数量    |
| --- | ------- |
| 训练集 | 600,000 |
| 验证集 | 15,000  |
| 测试集 | 15,000  |

数据样例（`dataset/samples.jsonl`）：

```json
{
  "review_body": "包装太水了，就一个塑料袋，对于爱书如命的我受不了。纸张较差~",
  "review_title": "纸张差"
}
```

* `review_body`：用户评论正文
* `review_title`：评论标题

---

## 2. 模型权重下载

项目使用预训练模型 `Qwen/Qwen3-0.6B`，方便快速学习，也可根据算力选择更大模型。

```bash
python3 download_model.py
```

模型文件结构示例：

```
├── LICENSE
├── README.md
├── config.json
├── configuration.json
├── generation_config.json
├── merges.txt
├── model.safetensors      # 模型权重 (~1.5G)
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json
```

> `model.safetensors` 为模型权重，其余文件用于 tokenizer 配置和模型生成参数。

---

## 3. 模型微调

此脚本用于对大语言模型进行LoRA微调，支持通过命令行参数灵活调整训练超参数。

### 运行方式

```bash
python3 fintune_llm.py \
    --model_path ./model_weights/Qwen/Qwen3-0.6B \
    --data_path ./dataset/train.jsonl \
    --output_dir ./output \
    --batch_size 12 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --epochs 30 \
    --save_steps 100 \
    --learning_rate 1e-4 \
    --sample_size 5000 \
    --use_lora True
```

### 参数说明

| 参数                              | 类型    | 默认值                               | 说明               |
| ------------------------------- | ----- | --------------------------------- | ---------------- |
| `--model_path`                  | str   | `./model_weights/Qwen/Qwen3-0.6B` | 预训练模型路径          |
| `--data_path`                   | str   | `./dataset/train.jsonl`           | 训练数据路径           |
| `--output_dir`                  | str   | `./output`                        | 模型保存路径           |
| `--batch_size`                  | int   | 12                                | 每个设备训练的batch大小   |
| `--gradient_accumulation_steps` | int   | 4                                 | 梯度累积步数，模拟更大batch |
| `--logging_steps`               | int   | 10                                | 日志记录步数，用于监控训练状态  |
| `--epochs`                      | int   | 3                                 | 训练轮数             |
| `--save_steps`                  | int   | 100                               | 保存检查点间隔          |
| `--learning_rate`               | float | 1e-4                              | 学习率              |
| `--sample_size`                 | int   | None                              | 数据采样大小，可选        |
| `--use_lora`                    | bool  | True                              | 是否启用LoRA微调       |


* 可根据显存调整 `batch_size` 和 `gradient_accumulation_steps`。
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:CE:00.0 Off |                  Off |
|  0%   36C    P2            194W /  450W |   13957MiB /  24564MiB |     67%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            3344      C   python3                               13948MiB |
+-----------------------------------------------------------------------------------------+

### 实际运行情况

#### LoRA 参数配置
- **batch size**: 8  
- **gradient accumulation steps**: 1  
- **target modules**:  
  ```text
  q_proj, k_proj, v_proj, o_proj,
  gate_proj, up_proj, down_proj
    ```

* **LoRA rank (r)**: 8  — 平衡效果与效率
* **LoRA alpha**: 32  — 缩放因子（建议设置为 2*r）
* **LoRA dropout**: 0.1  — 正则化比率

#### 硬件与训练性能

* **显卡**: RTX 4090（一共24GB)
* **显存占用**: 14.7 GB
* **训练速度**: 1.1 step/s
* **五分钟训练步数**: ~300 steps
* **Loss 收敛**: 约 2000 steps 基本收敛，到2.0左右


---

## 4. 模型推理

### 运行方式

```bash
python inference.py \
    --lora_path "output/checkpoint-315" \
    --review_body "键盘还可以，比较舒适，但是鼠标里面两节电池让人很难受，因为鼠标得经常移动，重量比较敏感，这个设计比较差。"
```


### 参数说明

| 参数                     | 类型    | 默认值                        | 说明                             |
| ---------------------- | ----- | -------------------------- | ------------------------------ |
| `--model_name`         | str   | `./models/Qwen/Qwen3-0.6B` | 模型路径                           |
| `--lora_path`          | str   | None                       | LoRA 权重路径（可选）                  |
| `--review_body`        | str   | 必填                         | 商品评论内容                         |
| `--max_new_tokens`     | int   | 64                         | 生成的最大 token 数                  |
| `--do_sample`          | bool  | False                      | 是否启用随机采样                       |
| `--top_p`              | float | 0.9                        | Nucleus采样概率                    |
| `--temperature`        | float | 0.7                        | 控制创造性                          |
| `--repetition_penalty` | float | 1.1                        | 重复惩罚系数                         |
| `--device`             | str   | auto                       | 模型加载设备，如 `cuda`, `cpu`, `auto` |

> 注意事项：
>
> 1. 使用 LoRA 权重需与基础模型匹配。
> 2. `do_sample=True` 时生成文本更具多样性，否则使用贪心解码。
> 3. 高 `temperature` 值可增加创造性，但可能导致不稳定输出。

---

## 5. 批量推理及评估

对整个测试集进行批量推理，并计算 ROUGE 评估指标。

### 运行方式

```bash
python3 inference_batch.py \
    --input ./dataset/test.jsonl \
    --output ./output/inference_result.jsonl \
    --lora_path ./output/checkpoint-315 \
    --model_name ./model_weights/Qwen/Qwen3-0.6B
```


### 参数说明

| 参数                     | 类型    | 默认值                               | 说明                                              |
| ---------------------- | ----- | --------------------------------- | ----------------------------------------------- |
| `--input`              | str   | `./dataset/test.jsonl`            | 输入 JSONL 文件，每行包含 `review_body` 和 `review_title` |
| `--output`             | str   | `inference_result.jsonl`          | 输出 JSONL 文件，每条记录增加 `summary` 字段                 |
| `--model_name`         | str   | `./model_weights/Qwen/Qwen3-0.6B` | 基础模型路径                                          |
| `--lora_path`          | str   | None                              | LoRA 权重路径（可选）                                   |
| `--batch_size`         | int   | 64                                | 一次推理的批量大小                                       |
| `--sample_size`        | int   | 640                               | 随机抽样验证条数                                        |
| `--max_new_tokens`     | int   | 64                                | 生成的最大 token 数                                   |
| `--min_new_tokens`     | int   | 0                                 | 生成的最小 token 数                                   |
| `--do_sample`          | bool  | False                             | 是否启用随机采样生成（启用时使用 `--do_sample`）                 |
| `--top_p`              | float | 0.9                               | top-p 采样的累积概率阈值                                 |
| `--temperature`        | float | 0.7                               | 控制生成文本的创造性                                      |
| `--repetition_penalty` | float | 1.1                               | 重复惩罚系数                                          |

---

### 输出示例

输出文件为 JSONL 格式，每条数据增加 `summary` 字段：

```json
{
    "review_body": "物流很快，包装完好。",
    "review_title": "满意的购物体验",
    "summary": "物流快，包装好，购物体验满意。"
}
```

终端输出 ROUGE 结果示例：

```
Sampled 1280 entries -- ROUGE-1 F1: 0.0573, ROUGE-L F1: 0.0554
```