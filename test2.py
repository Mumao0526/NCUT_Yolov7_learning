colors = ((0,165,255), (0,0,255), (0,255,0), (128,0,128), (255,255,0))  # 分別對應於橙色、紅色、綠色

def draw_box(img, box: tuple, label: str,
             box_color:tuple=(0,255,0), box_thick=1,
             font_face=cv2.FONT_HERSHEY_COMPLEX_SMALL,
             font_size=.6, font_color=(0,0,0), font_thick=1):
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
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, box_thick)
    (fw, fh), _ = cv2.getTextSize(label, font_face, font_size, font_thick)
    p = 1 # padding
    cv2.rectangle(img, (x1, y1), (x1 + fw + p*2, y1 + fh + p*2), box_color, -1)
    cv2.putText(img, label, (x1 + p, y1 + fh + p), font_face, font_size, font_color, font_thick)


def yolo2cvLabel(img_w, img_h, x_center, y_center, width, height):
    """
    將YOLO標籤格式轉換為OpenCV格式。

    參數:
    - img_w: 圖像的寬度
    - img_h: 圖像的高度
    - x_center: 標籤中心的x座標
    - y_center: 標籤中心的y座標
    - width: 標籤的寬度
    - height: 標籤的高度

    返回:
    - 矩形框的座標，格式為 (x1, y1, x2, y2)
    """
    x1 = int((x_center - width / 2) * img_w)
    y1 = int((y_center - height / 2) * img_h)
    x2 = int((x_center + width / 2) * img_w)
    y2 = int((y_center + height / 2) * img_h)
    return x1, y1, x2, y2


def mark_dataset(classes, src_db_dir: str, dest_db_dir: str, src_subpath=".bmp"):
    """
    從原始圖像文件讀取YOLO標記並在圖像上繪製標籤框，然後保存到新的目錄中。

    參數:
    - classes: 類別名稱的tuple
    - src_db_dir: 原始圖像的目錄路徑
    - dest_db_dir: 儲存繪製標籤後圖像的目錄路徑
    - src_subpath: 原始圖像文件的副檔名
    """
    src_path_len = len(src_db_dir)
    imgs = glob.glob(f"{src_db_dir}/*{src_subpath}", recursive=True)
    if not imgs:
        print(f"No images found in {src_db_dir} with extension {src_subpath}")
        return

    for src_img in imgs:
        # print(src_img)
        src_label = src_img[:-4] + ".txt" # 標記檔案路徑
        # print(src_label)
        dest_img_path = os.path.join(dest_db_dir, src_img[src_path_len:-4] + ".jpg")
        dest_img_dir = os.path.dirname(dest_img_path)
        if not os.path.exists(dest_img_dir):
            os.makedirs(dest_img_dir)
        # print(dest_img_path)

        img = cv2.imread(src_img)
        if img is None:
            print(f"Failed to read image {src_img}")
            continue

        img_h, img_w, _ = img.shape
        with open(src_label, "r") as file:
            for line in file.read().splitlines():
                # print(line)
                cls, rx, ry, rw, rh = [float(v) for v in line.split(" ")]
                cls = int(cls)
                x1, y1, x2, y2 = yolo2cvLabel(img_w, img_h, rx, ry, rw, rh)
                draw_box(img, (x1, y1, x2, y2), classes[cls], colors[cls])
            cv2.imwrite(dest_img_path, img)
            print(f"Image saved to {dest_img_path}")

mark_dataset(
    classes = classes,
    src_db_dir = "./animals/data/images-1-coverted",
    dest_db_dir = "./animals/data/images-2-marked"
)
