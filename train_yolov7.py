import glob
import re
import requests
import subprocess
import urllib3
import configparser
from pathlib import Path

urllib3.disable_warnings()

# 讀取 .ini 檔案
config = configparser.ConfigParser()
config.read('./config.ini')

# ------------------- 參數設定 -------------------
class_list = config.get('prepare', 'class_list').strip('[]').replace('"', '').split(', ')
yolov7_repo = config.get('urls', 'yolov7_repo')
pretrained_model = config.get('urls', 'pretrained_model')
python_path = config.get('training', 'python_path')
dataset_yaml = config.get('training', 'dataset_yaml')
device = config.get('training', 'device')
batch_size = config.get('training', 'batch_size')
epochs = config.get('training', 'epochs')
project_dir = config.get('training', 'project_dir')
# ----------------------------------------------

# 使用 pathlib 處理路徑
models_dir = Path("data/models")
models_dir.mkdir(parents=True, exist_ok=True)  # 確保目錄存在


def download_file(url, dest_path):
    response = requests.get(url, stream=True, verify=False)  # 警告：禁用 SSL 驗證
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def setup_environment():
    # 創建必要的目錄
    models_dir.mkdir(parents=True, exist_ok=True)

    # 克隆 YOLOv7 代碼庫
    yolov7_path = Path("./yolov7")
    if not yolov7_path.exists():
        subprocess.run(["git", "clone", yolov7_repo], check=True)

    # 修改 requirements.txt 文件中的指定部分
    requirements_path = yolov7_path / "requirements.txt"
    with requirements_path.open("r") as file:
        content = file.read()
    content = re.sub(r"(,<1\.24\.0)$", " #\\1", content)
    with requirements_path.open("w") as file:
        file.write(content)

    # 安裝依賴
    subprocess.run(["pip", "install", "-r", str(requirements_path)], check=True)

    # 使用 requests 下載模型文件
    download_file(pretrained_model, models_dir / "yolov7.pt")


def config_model_yaml(dest_models_dir: Path):
    model_cfg_path = Path("yolov7/cfg/training/")
    models = list(model_cfg_path.glob("yolov7*.yaml"))
    for model_path in models:
        with model_path.open("r") as f:
            text = f.read()
        # 需要先定義 `class_list`
        text = re.sub(r"^(nc\:\s*)\d+", r"\g<1>{}".format(len(class_list)), text, flags=re.M)
        dest_models_file = dest_models_dir / model_path.name
        with dest_models_file.open("w") as f:
            f.write(text)


def train_model():
    config_model_yaml(models_dir)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    subprocess.run([
        python_path, str(Path("./yolov7/train.py")),
        '--weights', str(models_dir / "yolov7.pt"),
        '--cfg', str(models_dir / "yolov7.yaml"),
        '--data', dataset_yaml,
        '--device', device, '--batch-size', batch_size, '--epochs', epochs,
        '--project', project_dir, '--name', f'{timestamp}-yolov7-gpu'
    ], check=True)


if __name__ == "__main__":
    setup_environment()
    train_model()
