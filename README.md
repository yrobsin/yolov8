# YOLOv8 图像分割与特征对比

## 项目简介
本项目旨在使用 YOLOv8 进行服装图像的自动分割与特征提取，进而实现相似度对比的功能。该系统可以帮助用户识别和比较不同服装图片中的相似元素，如颜色、形状、和款式细节。通过训练后的 YOLOv8 模型，项目能够自动分割图像中的服装部分，并提取其深度特征和颜色特征。随后，系统会将这些特征与数据库中的图像进行对比，找出相似度最高的几张图片，并返回结果。

该项目特别适合应用在电商平台、服装设计、智能搭配等场景中，能够帮助用户快速找到相似款式的服装，或者用于推荐系统、自动化设计分析等。无论是服装行业的从业者还是对时尚感兴趣的开发者，都能从本项目中获得有价值的工具和灵感。

## 文件结构
- `train.py`: 用于训练 YOLOv8 模型的脚本。
- `灰度处理第二版.py`: 实现图像分割和特征提取、比较的脚本。
- `runs/segment/`: 保存训练后的模型和结果。
- `images/`: 存放要进行特征提取和比较的图像。
- `features/`: 存放提取的深度特征和颜色特征。

## 环境配置
建议在带有 CUDA 的设备（如 GPU）上运行此项目，以获得更好的性能。如果没有 GPU，亦可在 CPU 上运行，但可能会较慢。

### 依赖库
请确保已安装以下 Python 库：
- `torch`
- `numpy`
- `opencv-python`
- `ultralytics`
- `matplotlib`
- `scikit-learn`

可以通过以下命令安装依赖：
```bash
pip install -r requirements.txt

### 第二部分
```markdown
## 使用说明

### 训练模型
1. 确保数据集已按照 `dataset.yaml` 文件中的配置进行组织。
2. 运行 `train.py` 文件来训练模型。训练好的模型将会保存在 `runs/segment/` 目录下。
```bash
python train.py

进行图像分割与比较
将要比较的图像存放在 images/ 目录中。
运行 灰度处理第二版.py 脚本。此脚本将提取图像的特征，并将与数据库中图像的特征进行比较，输出最相似的几张图片。

注意事项
请确保输入的图像格式为 .jpg 或 .png，否则可能无法正确处理。
### 第三部分
```markdown
## 示例代码

### 训练模型
```python
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 设置字体以支持中文显示
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

def train_model():
    model = YOLO("runs/segment/模型1.2/weights/best.pt")
    model.train(
        data="dataset.yaml", 
        epochs=100,  
        imgsz=640,
        lr0=0.004,
        lrf=0.0001,
        batch=12,
        verbose=False,
        plots=True
    )

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    train_model()
使用前建议先对图片进行预处理，确保它们的尺寸和格式适合 YOLOv8 模型的输入要求。

### 第四部分
```markdown
### 进行图像分割与比较
```python
import torch
import numpy as np
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

# 设置处理的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("runs/segment/模型1.1/weights/best.pt").to(device)

# 指定图片目录和特征保存目录
image_dir = "images"
feature_dir = "features"
os.makedirs(feature_dir, exist_ok=True)

fixed_feature_size = 2048

def extract_deep_features(model, img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_tensor = torch.tensor(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    with torch.no_grad():
        x = model.model.model[:10](img_tensor)
        deep_features = model.model.model[10](x).flatten().cpu().numpy()

    color_features = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_features = cv2.normalize(color_features, color_features).flatten()

    return deep_features, color_features

def process_image(img_name):
    if img_name.endswith(('.jpg', '.png')):
        img_path = os.path.join(image_dir, img_name)
        deep_features, color_features = extract_deep_features(model, img_path)
        deep_features = np.pad(deep_features, (0, max(0, fixed_feature_size - len(deep_features))), mode='constant')[:fixed_feature_size]
        color_features = np.pad(color_features, (0, max(0, fixed_feature_size - len(color_features))), mode='constant')[:fixed_feature_size]
        np.save(os.path.join(feature_dir, f"{os.path.splitext(img_name)[0]}_deep.npy"), deep_features)
        np.save(os.path.join(feature_dir, f"{os.path.splitext(img_name)[0]}_color.npy"), color_features)

with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(process_image, os.listdir(image_dir))

### 第五部分
```markdown
def compare_image_with_database(new_image_path, feature_dir, image_dir):
    deep_feature_files = [os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith('_deep.npy')]
    color_feature_files = [f.replace('_deep.npy', '_color.npy') for f in deep_feature_files]

    deep_features_db = [np.load(f) for f in deep_feature_files]
    color_features_db = [np.load(f) for f in color_feature_files]

    deep_features, color_features = extract_deep_features(model, new_image_path)
    deep_features = np.pad(deep_features, (0, max(0, fixed_feature_size - len(deep_features))), mode='constant')[:fixed_feature_size]
    color_features = np.pad(color_features, (0, max(0, fixed_feature_size - len(color_features))), mode='constant')[:fixed_feature_size]

    deep_similarities = cosine_similarity([deep_features], deep_features_db)
    color_similarities = cosine_similarity([color_features], color_features_db)
    similarities = (deep_similarities * deep_feature_weight) + (color_similarities * color_feature_weight)

    above_threshold_indices = np.where(similarities[0] > 0.8)[0]

    if len(above_threshold_indices) > 0:
        sorted_indices = np.argsort(-similarities[0, above_threshold_indices])
        top_indices = sorted_indices[:5]
        top_similarities = similarities[0, above_threshold_indices][top_indices]

        top_image_paths = []
        for i in above_threshold_indices[top_indices]:
            image_path = os.path.join(image_dir, os.path.basename(deep_feature_files[i]).replace('_deep.npy', '.jpg'))
            if not os.path.exists(image_path):
                image_path = image_path.replace('.jpg', '.png')
            top_image_paths.append(image_path)

    else:
        max_index = np.argmax(similarities[0])
        top_similarities = [similarities[0, max_index]]
        top_image_paths = []
        image_path = os.path.join(image_dir, os.path.basename(deep_feature_files[max_index]).replace('_deep.npy', '.jpg'))
        if not os.path.exists(image_path):
            image_path = image_path.replace('.jpg', '.png')
        top_image_paths.append(image_path)

    return top_image_paths, top_similarities

def display_images(image1_path, image_paths):
    plt.figure(figsize=(15, 10))
    image1 = cv2.imread(image1_path)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    plt.subplot(1, len(image_paths) + 1, 1)
    plt.title("Uploaded Image")
    plt.imshow(image1)
    plt.axis('off')

    for i, image_path in enumerate(image_paths, 2):
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.subplot(1, len(image_paths) + 1, i)
            plt.title(f"Similar Image {i-1}")
            plt.imshow(image)
            plt.axis('off')

    plt.show()

new_image_path = "images/2.jpg"
top_image_paths, top_similarities = compare_image_with_database(new_image_path, feature_dir, image_dir)

if len(top_similarities) > 0 and top_similarities[0] > 0.8:
    print(f"共有{len(top_similarities)}张图片相似度大于0.8，其中相似度最高达到了{round(top_similarities[0], 3)}")
else:
    print(f"没有图片相似度超过0.8，最高的相似度为{round(top_similarities[0], 3)}")

## 运行效果展示
<img width="392" alt="b5517393d150eedac8a5b647355bbd8" src="https://github.com/user-attachments/assets/55719298-995a-4491-ad2b-2c6fd365b2f9">

下图展示了使用 YOLOv8 进行服装图像分割和相似度对比的效果：
