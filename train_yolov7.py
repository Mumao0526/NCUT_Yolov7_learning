import glob
import re
import os
import requests
import subprocess
import urllib3
from pathlib import Path


urllib3.disable_warnings()

# 虛擬環境中 Python 的路徑
python_path = 'D:/USER/Coding/NCUT_Yolov7_learning/.venv/Scripts/python.exe'

# 定義需要的類別
classes = ("Rhino", "Buffalo", "Penguin", "Kangaroo", "Jaguar")


def download_file(url, dest_path):
    response = requests.get(url, stream=True, verify=False)  # 警告：禁用 SSL 驗證
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def setup_environment():
    models_dir = r"data\special_zoo-animals\models"
    # 創建必要的目錄
    os.makedirs(f"{models_dir}", exist_ok=True)

    # 克隆 YOLOv7 代碼庫
    if not os.path.exists("./yolov7"):
        subprocess.run(
            ["git", "clone", "https://github.com/WongKinYiu/yolov7"], check=True
        )

    # 修改 requirements.txt 文件中的指定部分
    with open("./yolov7/requirements.txt", "r") as file:
        content = file.read()
    content = re.sub(r"(,<1\.24\.0)$", " #\\1", content)
    with open("./yolov7/requirements.txt", "w") as file:
        file.write(content)

    # 安裝依賴
    subprocess.run(["pip", "install", "-r", "./yolov7/requirements.txt"], check=True)

    # # 下載預訓練的模型
    # subprocess.run(
    #     [
    #         "wget",
    #         "-P",
    #         "./zoo-animals/models/",
    #         "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
    #     ],
    #     check=True,
    # )

    # 使用 requests 下載模型文件
    download_file(
        "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
        f"{models_dir}/yolov7.pt",
    )


def config_model_yaml(dest_models_dir:str):
    """
    更新指定目錄下所有YOLOv7模型的配置文件，將類別數量(nc)設置為與全局`classes`變量一致。

    參數:
    - dest_models_dir: 包含YOLOv7模型配置文件的目錄路徑
    """
    # 搜索目錄下所有以'yolov7'開頭的YAML文件
    models = glob.glob(r"yolov7\cfg\training\yolov7*.yaml")
    text = ""
    for model in models:
        # 打開每個模型配置文件並讀取內容
        with open(model, "r") as f:
            text = f.read()
        # 使用正則表達式替換文件中的類別數量(nc)，設置為全局`classes`變量的長度
        text = re.sub(r"^(nc\:\s*)\d+", r"\g<1>{}".format(len(classes)), text, flags=re.M)
        dest_models_file = Path(dest_models_dir) / Path(model).name
        # 將修改後的內容寫回文件
        with open(dest_models_file, "w") as f:
            f.write(text)


def train_model():
    """
    執行模型訓練。
    """
    # 調用函數以更新配置文件
    config_model_yaml(r"data\special_zoo-animals\models")

    # 獲取當前時間戳
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    subprocess.run([
        python_path, './yolov7/train.py',
        '--weights', r'data\special_zoo-animals\models\yolov7.pt',
        '--cfg', r"data\special_zoo-animals\models\yolov7.yaml",
        '--data', r'data\special_zoo-animals\data\data-v7.yaml',
        '--device', '0', '--batch-size', '6', '--epochs', '80',
        '--project', './temp_formu', '--name', f'{timestamp}-yolov7-gpu'
    ], check=True)


if __name__ == "__main__":
    setup_environment()
    train_model()
