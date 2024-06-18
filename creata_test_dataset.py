# 測試用資料集
import shutil
import os
import random
from pathlib import Path

num_samples = 100  # 指定要取得的圖片數量
dataset_dir = r".\data\input_dataset\src_dataset\zoo-animals\data\input_dataset"
dest_dir_path = r".\data\src_dataset\test_dataset"
dataset_path = Path(dataset_dir)
extensions = [
    "*.JPG",
    "*.jpg",
    "*.png",
    "*.PNG",
    "*.bmp",
    "*.BMP",
    "*.jpeg",
    "*.JPEG",
]
imgs_path = [
    img_path for ext in extensions for img_path in dataset_path.rglob(ext)
]  # 取得資料集內所有圖片的路徑

r = [
    random.randint(0, len(imgs_path) - 1) for _ in range(num_samples)
]  # 產生隨機索引值
if not Path(dest_dir_path).exists():
    os.makedirs(dest_dir_path)

for i in r:
    fname = Path(imgs_path[i])
    # 計算目標圖片的路徑，保留子資料夾結構
    relative_path = os.path.relpath(imgs_path[i], dataset_dir)  # 取得相對路徑
    relative_dir = os.path.dirname(relative_path)  # 取得相對路徑的資料夾
    target_img_dir = os.path.join(dest_dir_path, relative_dir)
    if not Path(target_img_dir).exists():
        os.makedirs(target_img_dir)
    shutil.copyfile(imgs_path[i], os.path.join(dest_dir_path, relative_path))
    shutil.copyfile(Path(imgs_path[i]).with_suffix(".xml"), os.path.join(dest_dir_path, Path(relative_path).with_suffix(".xml")))
