from Dataset import Dataset

dataset_directory = r".\data\special_zoo-animals"   # 輸入資料集
classes = ("Penguin", "Kangaroo", "Rhino", "Jaguar", "Buffalo")
dataset = Dataset.Dataset_Loader(dataset_dir=dataset_directory, classes=classes)

label_dir = r".\data\special_zoo-animals\data\labels"
subset_names = ("train", "val", "test")
dataset.count_dataset(label_dir, subset_names)


'''
!mkdir -p {mygd}/temp_formu
!git clone https://github.com/WongKinYiu/yolov7
!sed -i "s/\(,<1\.24\.0\)$/ #\1/" ./yolov7/requirements.txt
!pip install -r ./yolov7/requirements.txt

!mkdir -p ./zoo-animals/models/
!cp ./yolov7/cfg/training/yolov7*.yaml ./zoo-animals/models/
!wget -P ./zoo-animals/models/ https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt


import glob
import re

classes = ("Rhino", "Buffalo", "Penguin", "Kangaroo", "Jaguar")

def config_model_yaml(dest_models_dir:str):
    """
    更新指定目錄下所有YOLOv7模型的配置文件，將類別數量(nc)設置為與全局`classes`變量一致。

    參數:
    - dest_models_dir: 包含YOLOv7模型配置文件的目錄路徑
    """
    # 搜索目錄下所有以'yolov7'開頭的YAML文件
    models = glob.glob(f"{dest_models_dir}/yolov7*.yaml")
    text = ""
    for model in models:
        # 打開每個模型配置文件並讀取內容
        with open(model, "r") as f:
            text = f.read()
        # 使用正則表達式替換文件中的類別數量(nc)，設置為全局`classes`變量的長度
        text = re.sub(r"^(nc\:\s*)\d+", r"\g<1>{}".format(len(classes)), text, flags=re.M)
        # 將修改後的內容寫回文件
        with open(model, "w") as f:
            f.write(text)

# 調用函數以更新配置文件
config_model_yaml("./zoo-animals/models/")


!python ./yolov7/train.py \
--weight /content/zoo-animals/models/yolov7.pt \
--cfg /content/zoo-animals/models/yolov7.yaml \
--data /content/zoo-animals/data/data-v7.yaml \
--device 0 --batch-size 24 --epoch 80 \
--project {mygd}/temp_formu \
--name $(date +%Y%m%dt%H%M%S)-yolov7-gpu
'''