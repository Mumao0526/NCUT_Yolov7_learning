import cv2
import os
import shutil
import random
import seaborn as sns
import matplotlib.pyplot as plt
from random import shuffle
from bs4 import BeautifulSoup as bs
from pathlib import Path


class Dataset_Loader:
    def __init__(
        self,
        dataset_dir: str,
        classes: tuple,
    ):
        """
        初始化Dataset類別。

        參數:
        - imgs_path: 圖像資料的路徑
        - labels_path: 圖像標籤的路徑
        - classes: 一個包含所有類別名稱的tuple
        """
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
        self.dataset_path = Path(dataset_dir)
        self.imgs_path = self.get_pathslist(
            self.dataset_path, extensions
        )  # 取得資料集內所有圖片的路徑
        extensions = ["*.xml", "*.XML"]
        self.labels_path = self.get_pathslist(
            self.dataset_path, extensions
        )  # 取得資料集內所有標籤的路徑
        self.classes = classes
        self.draw_colors = [
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(len(self.classes))
        ]  # 依照類別數量產生隨機顏色

    def run(
        self,
        dest_dir: str,
        subset_names: tuple,
        subset_percentages: tuple,
    ):
        """
        將資料集轉換成YOLO格式，並分割成訓練集、驗證集和測試集，最後生成data-v7.yaml配置文件。
        """
        dest_images_path = Path(os.path.join(dest_dir, "images"))
        dest_labels_path = Path(os.path.join(dest_dir, "labels"))

        dest_dir = Path(dest_dir)
        dest_lists_path = dest_dir

        self.show_dataset_info()
        self.splite_dataset_to_dir(
            dest_images_path, dest_lists_path, subset_names, subset_percentages
        )
        self.count_dataset(dest_labels_path, subset_names, show_plt=False)
        self.make_data_yaml(dest_dir)

        self.show_distribution_of_classes()
        self.show_image_with_marked()

    def get_pathslist(self, src_path, extensions):
        """
        獲取指定路徑下的所有文件路徑列表
        src_path: 指定路徑
        extensions: 文件副檔名

        return: 文件路徑列表
        """
        # 创建一个集合来存储唯一的路径字符串
        unique_paths = set()
        for ext in extensions:
            unique_paths.update(str(p) for p in Path(src_path).rglob(ext))

        # 将唯一的路径字符串转换回 Path 对象，并返回这些 Path 对象的列表
        return [Path(p) for p in unique_paths]

    def show_dataset_info(self):
        """
        顯示資料集的基本信息，包括圖像數量、類別數量和類別名稱。
        """
        print(f"Dataset path: {self.dataset_path}")
        print(f"Number of images: {self.get_dataset_size()}")
        print(f"Number of labels: {len(self.labels_path)}")
        print(f"Number of classes: {len(self.classes)}")
        print(f"Classes: {self.classes}")

    def show_distribution_of_classes(self):
        """
        顯示資料集中各個類別的分佈情況。
        """
        counts = [0] * len(self.classes)
        for txt in self.labels_path:
            if txt.exists():  # 檢查標記檔案是否存在
                with open(str(txt), "r") as file:
                    # 取得各個類別物件的個數
                    for line in file.readlines():
                        cls = int(line.split(" ")[0])
                        counts[cls] += 1
            else:
                print(f"Warning: Label file not found for {txt}")
        print("Class distribution:")
        for i, cls in enumerate(self.classes):
            print(f"{cls}: {counts[i]}")

        # 以plt.bar()繪製長條圖
        sns.barplot(x=self.classes, y=counts, hue=self.classes, alpha=0.8, palette="rocket", legend=False)
        plt.title("Distribution of Class in Image Dataset", fontsize=16)
        plt.xlabel("Class", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.xticks(rotation=45)
        plt.show()

    def show_image_with_marked(self, num_samples=16):
        """
        顯示16張圖片及其標籤。
        """
        random_index = [
            random.randint(0, len(self.imgs_path) - 1) for _ in range(num_samples)
        ]  # 產生隨機索引值

        fig, axes = plt.subplots(
            nrows=int(num_samples**0.5), ncols=int(num_samples**0.5), figsize=(10, 10), subplot_kw={"xticks": [], "yticks": []}
        )

        def generate_label_path(img_path):
            """
            生成標記檔案的路徑，且適用於Windows、Linux作業環境。
            """
            img_path = Path(img_path)
            parts = list(img_path.parts)

            # 找到 'images' 的位置并替换为 'labels'
            try:
                index = parts.index("images")
                parts[index] = "labels"
            except ValueError:
                raise ValueError(
                    "Error: 'images' not found in image path, cant find label file."
                )

            # 重新构建路径，并更改文件的后缀
            new_path = Path(*parts).with_suffix(".txt")

            return new_path

        for i, ax in enumerate(axes.flat):
            img_path = self.imgs_path[random_index[i]]
            label_path = generate_label_path(img_path)  # 標記檔案路徑
            img = cv2.imread(str(img_path))

            if img is None:
                print(f"Failed to read image for display {img_path}")
                continue

            img_h, img_w, _ = img.shape

            if Path(label_path).exists():  # 檢查標記檔案是否存在
                with open(label_path, "r") as file:
                    for line in file.read().splitlines():
                        cls, rx, ry, rw, rh = [float(v) for v in line.split(" ")]
                        cls = int(cls)
                        x1, y1, x2, y2 = self._convert_label_bbox_to_cv_format(
                            img_w, img_h, rx, ry, rw, rh
                        )
                        self.draw_box(
                            img,
                            (x1, y1, x2, y2),
                            self.classes[cls],
                            self.draw_colors[cls],
                        )
            else:
                print(f"Warning: Label file not found for {label_path}")
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        plt.tight_layout()
        plt.show()

    def get_dataset_size(self):
        return len(self.imgs_path)

    def _convert_label_bbox_to_yolo_format(
        self, img_w: int, img_h: int, x1: int, y1: int, x2: int, y2: int
    ):
        """
        將OpenCV格式的標籤轉換為YOLO格式的標籤。

        參數:
        - img_w: 圖像的寬度
        - img_h: 圖像的高度
        - x1, y1: 矩形左上角的座標
        - x2, y2: 矩形右下角的座標

        返回:
        - (r_x, r_y, r_w, r_h): YOLO格式的標籤，包括物體中心的相對座標 (r_x, r_y) 和相對尺寸 (r_w, r_h)
        """
        # 計算矩形的寬度和高度
        w = x2 - x1
        h = y2 - y1
        # 計算矩形中心點的相對座標
        r_x = (x1 + w / 2) / img_w
        r_y = (y1 + h / 2) / img_h
        # 計算矩形寬度和高度的相對尺寸
        r_w = w / img_w
        r_h = h / img_h
        # 返回YOLO格式的標籤
        return r_x, r_y, r_w, r_h

    def _convert_label_bbox_to_cv_format(
        self, img_w: int, img_h: int, r_x: float, r_y: float, r_w: float, r_h: float
    ):
        """
        將YOLO格式的標籤轉換為OpenCV格式的標籤。

        參數:
        - img_w: 圖像的寬度
        - img_h: 圖像的高度
        - r_x, r_y: 物體中心的相對座標
        - r_w, r_h: 物體的相對寬度和高度

        返回:
        - (int(x1), int(y1), int(x2), int(y2)): OpenCV格式的標籤，包括左上角 (x1, y1) 和右下角 (x2, y2) 的座標
        """
        # 根據相對尺寸計算實際的矩形寬度和高度
        w = r_w * img_w
        h = r_h * img_h
        # 計算矩形左上角的座標
        x1 = r_x * img_w - round(w / 2)
        y1 = r_y * img_h - round(h / 2)
        # 計算矩形右下角的座標
        x2 = x1 + w
        y2 = y1 + h
        # 返回OpenCV格式的標籤
        return int(x1), int(y1), int(x2), int(y2)

    def label_file_voc2yolo(self, voc_file: str, yolo_file: str):
        """
        將VOC格式的XML標記文件轉換為YOLO格式的標記文件。

        參數:
        - classes: 一個包含所有類別名稱的tuple
        - voc_file: VOC格式的XML檔案路徑
        - yolo_file: 輸出的YOLO格式標記文件路徑

        返回:
        - 無返回值，但會將轉換的結果寫入yolo_file
        """

        out = ""

        # 檢查標記檔案是否存在
        if not os.path.isfile(voc_file):
            print(f"Error: Failed to read voc file {voc_file}")
            return

        with open(voc_file, "r") as xml:
            content = xml.read()
            sp = bs(content, "xml")
            img_w = float(sp.find("width").text)
            img_h = float(sp.find("height").text)

            boxes = sp.find_all("object")
            for box in boxes:
                class_name = box.find("name").text
                if class_name not in self.classes:
                    print(
                        f"Warning: Class '{class_name}' not found in defined classes."
                    )
                    continue  # 跳過未定義的類別

                x1 = float(box.find("xmin").text)
                y1 = float(box.find("ymin").text)
                x2 = float(box.find("xmax").text)
                y2 = float(box.find("ymax").text)

                classes_index = self.classes.index(class_name)
                x, y, w, h = self._convert_label_bbox_to_yolo_format(
                    img_w, img_h, x1, y1, x2, y2
                )  # 轉成Yolo的格式
                out += f"{classes_index} {x} {y} {w} {h}\n"

        with open(yolo_file, "a") as txt:
            txt.write(out)

    def draw_box(
        self,
        img,
        box: tuple,
        label: str,
        box_color: tuple = (0, 255, 0),
        box_thick=3,
        font_face=cv2.FONT_HERSHEY_COMPLEX_SMALL,
        font_size=0.6,
        font_color=(0, 0, 0),
        font_thick=3,
    ):
        """
        在圖像上繪製一個包含文字標籤的矩形框。

        參數:
        - img: 要繪製的圖像
        - box: 矩形框的座標，格式為 (x1, y1, x2, y2)
        - label: 矩形框內顯示的標籤文字
        - box_color: 矩形框的顏色
        - box_thick: 矩形框的線條粗細
        - font_face: 文字的字體
        - font_size: 文字的大小
        - font_color: 文字的顏色
        - font_thick: 文字的線條粗細
        """
        if img is None:
            print("Failed to read image for drawing box.")
            return

        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, box_thick)
        (fw, fh), _ = cv2.getTextSize(label, font_face, font_size, font_thick)
        p = 1  # padding
        cv2.rectangle(img, (x1, y1), (x1 + fw + p * 2, y1 + fh + p * 2), box_color, -1)
        cv2.putText(
            img,
            label,
            (x1 + p, y1 + fh + p),
            font_face,
            font_size,
            font_color,
            font_thick,
        )

    def mark_dataset(self, src_db_dir: str, dest_db_dir: str, src_subpath=".bmp"):
        """
        從原始圖像文件讀取YOLO標記並在圖像上繪製標籤框，然後保存到新的目錄中。

        參數:
        - classes: 類別名稱的tuple
        - src_db_dir: 原始圖像的目錄路徑
        - dest_db_dir: 儲存繪製標籤後圖像的目錄路徑
        - src_subpath: 原始圖像文件的副檔名
        """

        for i in range(self.get_dataset_size()):
            src_label = Path(self.imgs_path[i]).with_suffix(".txt")  # 標記檔案路徑
            fname = Path(self.imgs_path[i]).stem  # 只取檔案名稱，不包含副檔名
            dest_img_path = os.path.join(dest_db_dir, fname + ".png")

            img = cv2.imread(str(self.imgs_path[i]))
            img_h, img_w, _ = img.shape

            if os.path.isfile(src_label):
                with open(src_label, "r") as file:
                    for line in file.read().splitlines():
                        cls, rx, ry, rw, rh = [float(v) for v in line.split(" ")]
                        cls = int(cls)
                        x1, y1, x2, y2 = self._convert_label_bbox_to_cv_format(
                            img_w, img_h, rx, ry, rw, rh
                        )
                        self.draw_box(
                            img,
                            (x1, y1, x2, y2),
                            self.classes[cls],
                            self.draw_colors[cls],
                        )
                    cv2.imwrite(dest_img_path, img)
            else:
                print(f"Warning: Label file not found to mark for {src_label}")

    def img_convert_to_bmp(self, src_img_path: str, dest_img_path: str):
        """
        將圖像轉換為BMP格式。

        參數:
        - src_img_path: 原始圖像文件的路徑
        - dest_img_path: 轉換後圖像文件的路徑
        """
        file_extension = Path(src_img_path).suffix
        if file_extension.lower() == ".bmp":
            shutil.copyfile(src_img_path, dest_img_path)  # 複製到指定路徑
        else:
            # 若非 BMP 格式，需讀取圖像並儲存為 BMP 格式。
            image = cv2.imread(str(src_img_path))

            if image is None:
                print(f"Failed to read image {src_img_path}")
                return None  # 無法讀取圖片，則退出

            cv2.imwrite(dest_img_path, image)  # 儲存為 BMP 格式

    def copy_to_subset_dir(
        self,
        dset_images_dir: str,
        dest_lists_dir: str,
        subset_name: str,
        subset_images: list,
        index_start: int,
    ):
        """
        將指定的圖像及其標籤文件從來源目錄複製到指定的子集目錄中，並記錄這些圖像的路徑。

        參數:
        - dset_images_dir: 存放子集的目標目錄
        - dest_lists_dir: 存放路徑列表文件的目錄
        - subset_name: 子集的名稱
        - subset_images: 要複製的圖像路徑列表
        - index_start: 計數的起始索引值
        """
        print(f"Copying images and label files to {subset_name} folder ...")  # 進度提示

        dset_labels_dir = os.path.join(dest_lists_dir, "labels")
        if not os.path.exists(dset_labels_dir):
            os.makedirs(dset_labels_dir)

        dest_imgs_dir_path = os.path.join(dset_images_dir, subset_name)
        dest_labels_dir_path = os.path.join(dset_labels_dir, subset_name)

        path_list = ""
        for img_path in subset_images:
            # 搬到分割資料夾，順便轉換成Yolov7接受格式

            # 計算目標圖片的路徑，保留子資料夾結構
            relative_path = os.path.relpath(img_path, self.dataset_path)  # 取得相對路徑
            relative_dir = os.path.dirname(relative_path)  # 取得相對路徑的資料夾
            target_img_dir = os.path.join(dest_imgs_dir_path, relative_dir)
            target_label_dir = os.path.join(dest_labels_dir_path, relative_dir)

            bmp_name = Path(img_path).stem + ".bmp"
            label_path = Path(img_path).with_suffix(".xml")
            target_img_path = os.path.join(target_img_dir, bmp_name)
            target_label_path = os.path.join(
                target_label_dir, Path(label_path).stem + ".txt"
            )

            # -- 複製圖片 --
            # 若目標資料夾不存在，則建立目標資料夾
            if not Path(target_img_dir).exists():
                os.makedirs(target_img_dir)
            if not Path(target_label_dir).exists():
                os.makedirs(target_label_dir)

            if not Path(img_path).exists():  # 檢查圖片是否存在
                print(f"Error: Failed to read image {img_path} for copying.")
                continue
            if not Path(label_path).exists():  # 檢查標籤是否存在
                print(f"Error: Failed to read label file {label_path} for copying.")
                continue

            self.standardize_img(
                img_path, target_img_path
            )  # 轉換成 BMP 格式。 （此舉可不做，但如此可以避免Yolo跳出警告）

            self.standardize_label(label_path, target_label_path)  # 轉換標籤文件格式

            path_list += str(target_img_path) + "\n"
            self.imgs_path[index_start] = target_img_path  # 更新圖片路徑
            index_start += 1  # 更新索引值

        with open(f"{dest_lists_dir}/{subset_name}_list.txt", "a") as txt:
            txt.write(path_list)

    def standardize_img(self, src_img_path, dest_img_path):
        """
        標準化圖片格式。
        """
        # 轉換成 BMP 格式。 （此舉可不做，但如此可以避免Yolo跳出警告）
        # self.img_convert_to_bmp(src_img_path, dest_img_path)
        shutil.copyfile(src_img_path, dest_img_path)

    def standardize_label(self, src_label_path, dest_label_path):
        """
        標準化標籤格式。
        """
        # 轉換VOC標籤文件格式成YOLO格式
        self.label_file_voc2yolo(src_label_path, dest_label_path)

    def splite_dataset_to_dir(
        self,
        dset_images_dir: str,
        dest_lists_dir: str,
        subset_names: tuple,
        subset_percentages: tuple,
    ):
        """
        將圖像數據集根據指定的百分比分割成多個子集。

        參數:
        - dset_images_dir: 存放生成的子集的目標目錄
        - dest_lists_dir: 存放子集路徑列表的目錄
        - subset_names: 分割子集的名稱組成的tuple
        - subset_percentages: 每個分割子集分配的圖像百分比組成的tuple
        """
        print("Splitting dataset into subsets ...")  # 進度提示

        if not os.path.exists(dset_images_dir):
            os.makedirs(dset_images_dir)
        if not os.path.exists(dest_lists_dir):
            os.makedirs(dest_lists_dir)

        imgs_len = self.get_dataset_size()
        shuffle(self.imgs_path)  # (***) 必須打亂順序

        # 複製到各個分割子集資料夾
        subset_begin = 0
        for i, subset_name in enumerate(subset_names):
            if i < len(subset_percentages) - 1:
                subset_percentage = subset_percentages[i]
                subset_num = int(imgs_len * subset_percentage / 100)  # 資料集圖片數量
                subset_end = subset_begin + subset_num
            else:
                subset_end = imgs_len

            self.copy_to_subset_dir(
                dset_images_dir,
                dest_lists_dir,
                subset_name,
                subset_images=self.imgs_path[subset_begin:subset_end],
                index_start=subset_begin,
            )
            subset_begin = subset_end

    def count_dataset(self, src_labels_dir, subset_names, show_plt=True):
        """
        統計指定目錄下各個子目錄（代表不同的子集）中的圖像數量及各類別的數量。

        參數:
        - src_images_dir: 存放子集的圖像目錄路徑

        輸出:
        - 打印出每個子集的圖像總數以及每個類別的數量
        """
        print("-"*10 + "Count dataset" + "-"*10)  # 進度提示

        extensions = ["*.txt", "*.TXT"]
        self.labels_path = self.get_pathslist(
            src_labels_dir, extensions
        )  # 取得資料集內所有標籤的路徑

        print(f"Number of labels: {len(self.labels_path)}")

        counts = {}  # 初始化一個字典來存放結果
        for txt in self.labels_path:
            # 判斷子集名稱
            set_name = None
            for subset_name in subset_names:
                if subset_name in txt.parts:
                    set_name = subset_name
                    break

            if set_name is None:
                print(f"Warning: Subset name not found in path {txt}")
                continue

            if set_name not in counts:  # 初始化子集的計數
                counts[set_name] = [0] * (len(self.classes) + 1)

            counts[set_name][0] += 1  # 為該子集的圖像數增加一

            if not Path(txt).exists():  # 檢查標記檔案是否存在
                print(f"Failed to read label file {txt} to count.")
                continue
            with open(txt, "r") as file:
                for line in file.readlines():
                    cls = int(line.split(" ")[0])  # 從每一行中讀取類別編號
                    counts[set_name][cls + 1] += 1  # 根據類別編號更新計數

        print("set: ", self.classes)  # 打印標題
        print(counts)  # 打印統計結果

        if show_plt:
            # 以plt.bar()繪製長條圖
            for set_name, numbers in counts.items():
                plt.figure(figsize=(10, 5))
                categories = [f"{cls}" for i, cls in enumerate(self.classes, 1) if i > 0]
                values = numbers[1:]  # Skip the total count
                sns.barplot(x=categories, y=values, hue=categories, alpha=0.8, palette="rocket", legend=False)
                plt.title(f"Data distribution in {str(set_name).upper()}")
                plt.ylabel("Counts")
                plt.xlabel("Classes")
                plt.show()

        print("-"*10 + "Count end" + "-"*10)  # 進度提示

    def make_data_yaml(self, dest_data_dir: str):
        """
        創建一個YAML格式的配置文件，指定用於YOLOv7模型訓練的數據集路徑，包括訓練集、驗證集和測試集。
        同時配置類別數量和類別名稱。

        參數:
        - dest_data_dir: 配置文件將被存放的目標目錄路徑
        """
        dest_data_dir = str(dest_data_dir).replace("\\", "/")
        # 配置文件內容，指定訓練集、驗證集、測試集的路徑，以及類別的數量和名稱
        yaml_content = f"""train: {dest_data_dir}/train_list.txt
val: {dest_data_dir}/val_list.txt
test: {dest_data_dir}/test_list.txt

nc: {len(self.classes)}

names: {list(self.classes)}
"""
        # 寫入文件
        with open(f"{dest_data_dir}/data-v7.yaml", "w") as file:
            file.write(yaml_content)
