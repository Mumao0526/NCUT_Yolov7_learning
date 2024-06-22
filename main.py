import os
from pathlib import Path
import shutil
from Dataset import Dataset
from configparser import ConfigParser

config = ConfigParser()
config.read("config.ini", encoding="utf-8")

# ------------------- 參數設定 -------------------
src_dataset_folder = config.get("prepare_dataset", "src_dataset_folder")
dest_dataset_folder = config.get("prepare_dataset", "dest_dataset_folder")
class_list = config.get("prepare_dataset", "classList").strip("[]").replace('"', "").split(", ")
subset_names = (
    config.get("prepare_dataset", "subset_names").strip("[]").replace('"', "").split(", ")
)
subset_percentages = tuple(
    map(int, config.get("prepare_dataset", "subset_percentages").strip("[]").split(", "))
)
is_show_chart = config.getboolean("prepare_dataset", "is_show_chart")
# ----------------------------------------------

# 確保目錄下沒有檔案
dest_path = Path(dest_dataset_folder)
if dest_path.exists():
    for item in dest_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)  # 刪除資料夾
        else:
            item.unlink()  # 刪除檔案
os.makedirs(dest_dataset_folder, exist_ok=True)  # 建立資料夾

# 創建資料集物件
dataset = Dataset.Dataset_Loader(dataset_dir=src_dataset_folder, classes=class_list)

# 抽樣檢查資料集的ELA變化
dataset.show_elv_sample()

# 運行數據處理
dataset.run(
    dest_dir=dest_dataset_folder,
    subset_names=subset_names,
    subset_percentages=subset_percentages,
    isShowChart=is_show_chart,
)
