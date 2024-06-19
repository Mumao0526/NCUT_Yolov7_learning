import os
from pathlib import Path
import shutil
from Dataset import Dataset

# 输入与输出资料集定义
dataset_directory = [
    r"data\input_dataset\penguin",
    r"data\input_dataset\kangaroo",
    r"data\input_dataset\buffalo",
    r"data\input_dataset\rhino",
    r"data\input_dataset\jaguar",
]

destination_dir = [
    r"data\special_zoo-animals\data",
]
classes = ("Penguin", "Kangaroo", "Rhino", "Jaguar", "Buffalo")

# 创建数据加载器实例
dataset_loaders = [
    Dataset.Dataset_Loader(dataset_dir=Path(dir), classes=classes) for dir in dataset_directory
]

# 处理每个目标目录
for i, dest_path in enumerate(destination_dir):
    dest_path = Path(dest_path)
    # 检查并准备目录
    if dest_path.exists():
        # 清空目录内容
        for item in dest_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        # 创建目录
        os.makedirs(dest_path, exist_ok=True)


for i in range(len(dataset_directory)):
    # 运行数据加载器
    # 如果是水牛数据集，只有训练集和测试集
    subset_names = ("train", "val", "test") if Path(dataset_directory[i]).stem != "buffalo" else ("train", "test")
    subset_percentages = (80, 10, 10) if Path(dataset_directory[i]).stem != "buffalo" else (80, 20)
    dataset_loaders[i].run(
        dest_dir=destination_dir[0],
        subset_names=subset_names,
        subset_percentages=subset_percentages
    )
