import os
import shutil
import pandas as pd
import json
import kagglehub

# 1. 下载 Kaggle 数据集
# 网站：https://www.kaggle.com/datasets/mexwell/amazon-reviews-multi/data
dataset_path = kagglehub.dataset_download("mexwell/amazon-reviews-multi")
print("Downloaded dataset path:", dataset_path)

# 2. 设置 dataset 目标文件夹
dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)

# 3. 移动文件到 dataset 文件夹
for file_name in os.listdir(dataset_path):
    src_file = os.path.join(dataset_path, file_name)
    dst_file = os.path.join(dataset_dir, file_name)
    shutil.move(src_file, dst_file)

print("Moved files to dataset folder:", dataset_dir)

# 4. 遍历 CSV 文件并处理
for file_name in os.listdir(dataset_dir):
    if file_name.endswith(".csv"):
        csv_file = os.path.join(dataset_dir, file_name)
        file_name = file_name.replace(".csv", "")
        jsonl_file = os.path.join(dataset_dir, f"{file_name}.jsonl")
        print("Processing:", csv_file)
        
        # 读取 CSV
        df = pd.read_csv(csv_file)
        
        # 筛选指定语言，我们这里只使用了英语、中文、日语，三种语言。
        df = df[df['language'].isin(['en', 'zh', 'ja'])]
        
        # 只保留 review_body 和 review_title
        df = df[['review_body', 'review_title']]
        
        # 输出到 jsonl
        with open(jsonl_file, 'w', encoding='utf-8') as f_out:
            for _, row in df.iterrows():
                json_line = json.dumps({
                    "review_body": str(row['review_body']),
                    "review_title": str(row['review_title'])
                }, ensure_ascii=False)
                f_out.write(json_line + "\n")
        
        print("Saved filtered JSONL to:", jsonl_file)

print("All CSV files processed.")
