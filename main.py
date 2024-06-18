import os
from pathlib import Path
import shutil
from Dataset import Dataset

dataset_directory = r".\data\test_dataset"    # 測試用
# dataset_directory = r".\data\input_dataset"   # 輸入資料集
classes = ("Penguin", "Kangaroo", "Rhino", "Jaguar", "Buffalo")
dataset = Dataset.Dataset_Loader(dataset_dir=dataset_directory, classes=classes)

# 指定目標目錄、子集名稱和各子集的百分比
destination_dir = r".\data\zoo-animals\data"

# 使用 pathlib 来处理路径
dest_path = Path(destination_dir)

# 检查目标目录是否存在
if dest_path.exists():
    # 如果目标目录存在，删除目录下的所有文件和子目录
    for item in dest_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)  # 删除目录
        else:
            item.unlink()  # 删除文件

os.makedirs(destination_dir, exist_ok=True)
subset_names = ("train", "val", "test")
subset_percentages = (80, 10, 10)  # 這表示訓練集80%，驗證集10%，測試集10%

# 運行數據處理
dataset.run(
    dest_dir=destination_dir,
    subset_names=subset_names,
    subset_percentages=subset_percentages,
)
