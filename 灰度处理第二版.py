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

# 加载训练后的模型到GPU或CPU上
model = YOLO("D:/Program Files/pythonProject/runs/segment/模型1.1/weights/best.pt")
model.to(device)

# 指定图片目录和特征保存目录
image_dir = "D:/Program Files/pythonProject/images"  # 替换为你的图片路径
feature_dir = "D:/Program Files/pythonProject/features"  # 特征文件保存路径
os.makedirs(feature_dir, exist_ok=True)

# 定义特征向量的固定大小
fixed_feature_size = 2048  # 减小深度特征的大小以减少负担

# 定义特征权重
deep_feature_weight = 0.8  # 深度特征的权重
color_feature_weight = 0.2  # 颜色特征的权重

# 提取深层特征的函数，不考虑颜色
def extract_deep_features(model, img_path):
    img = cv2.imread(img_path)

    # 提取灰度特征并降采样
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))  # 降低图片尺寸减轻计算量
    img_tensor = torch.tensor(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    with torch.no_grad():
        x = model.model.model[:10](img_tensor)
        deep_features = model.model.model[10](x).flatten().cpu().numpy()

    # 提取颜色特征
    color_features = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_features = cv2.normalize(color_features, color_features).flatten()

    # 返回深度特征和颜色特征的组合
    return deep_features, color_features

# 使用多线程并行处理图片
def process_image(img_name):
    if img_name.endswith(('.jpg', '.png')):
        img_path = os.path.join(image_dir, img_name)

        # 提取深层和颜色特征
        deep_features, color_features = extract_deep_features(model, img_path)

        # 如果特征向量长度不足，填充0；如果超出，截断
        deep_features = np.pad(deep_features, (0, max(0, fixed_feature_size - len(deep_features))), mode='constant')[:fixed_feature_size]
        color_features = np.pad(color_features, (0, max(0, fixed_feature_size - len(color_features))), mode='constant')[:fixed_feature_size]

        # 保存深层和颜色特征
        np.save(os.path.join(feature_dir, f"{os.path.splitext(img_name)[0]}_deep.npy"), deep_features)
        np.save(os.path.join(feature_dir, f"{os.path.splitext(img_name)[0]}_color.npy"), color_features)

# 使用线程池并行处理
with ThreadPoolExecutor(max_workers=4) as executor:  # 根据你的CPU核心数调整线程数
    executor.map(process_image, os.listdir(image_dir))

print("Feature extraction completed.")

def compare_image_with_database(new_image_path, feature_dir, image_dir):
    # 加载所有特征向量
    deep_feature_files = [os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith('_deep.npy')]
    color_feature_files = [f.replace('_deep.npy', '_color.npy') for f in deep_feature_files]

    deep_features_db = [np.load(f) for f in deep_feature_files]
    color_features_db = [np.load(f) for f in color_feature_files]

    # 计算新图片的特征向量，不考虑颜色
    deep_features, color_features = extract_deep_features(model, new_image_path)

    # 如果特征向量长度不足，填充0；如果超出，截断
    deep_features = np.pad(deep_features, (0, max(0, fixed_feature_size - len(deep_features))), mode='constant')[:fixed_feature_size]
    color_features = np.pad(color_features, (0, max(0, fixed_feature_size - len(color_features))), mode='constant')[:fixed_feature_size]

    # 计算深度特征的相似度
    deep_similarities = cosine_similarity([deep_features], deep_features_db)

    # 计算颜色特征的相似度
    color_similarities = cosine_similarity([color_features], color_features_db)

    # 加权组合深度特征和颜色特征的相似度
    similarities = (deep_similarities * deep_feature_weight) + (color_similarities * color_feature_weight)

    # 获取相似度大于阈值的图片，并排序
    above_threshold_indices = np.where(similarities[0] > 0.8)[0]

    if len(above_threshold_indices) > 0:
        sorted_indices = np.argsort(-similarities[0, above_threshold_indices])
        top_indices = sorted_indices[:5]
        top_similarities = similarities[0, above_threshold_indices][top_indices]

        top_image_paths = []
        for i in above_threshold_indices[top_indices]:
            image_path = os.path.join(image_dir, os.path.basename(deep_feature_files[i]).replace('_deep.npy', '.jpg'))
            # 如果 .jpg 文件不存在，尝试 .png
            if not os.path.exists(image_path):
                image_path = image_path.replace('.jpg', '.png')
            top_image_paths.append(image_path)

    else:
        # 如果没有任何图片的相似度超过0.8，只保留相似度最高的一张图片
        max_index = np.argmax(similarities[0])
        top_similarities = [similarities[0, max_index]]

        top_image_paths = []
        image_path = os.path.join(image_dir, os.path.basename(deep_feature_files[max_index]).replace('_deep.npy', '.jpg'))
        # 如果 .jpg 文件不存在，尝试 .png
        if not os.path.exists(image_path):
            image_path = image_path.replace('.jpg', '.png')
        top_image_paths.append(image_path)

    return top_image_paths, top_similarities

def display_images(image1_path, image_paths):
    # 读取并显示图片
    plt.figure(figsize=(15, 10))

    # 显示上传的图片
    image1 = cv2.imread(image1_path)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    plt.subplot(1, len(image_paths) + 1, 1)
    plt.title("Uploaded Image")
    plt.imshow(image1)
    plt.axis('off')

    # 显示相似图片
    for i, image_path in enumerate(image_paths, 2):
        # 检查文件是否存在
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.subplot(1, len(image_paths) + 1, i)
            plt.title(f"Similar Image {i-1}")
            plt.imshow(image)
            plt.axis('off')
        else:
            print(f"Image {image_path} not found, skipping.")

    plt.show()

# 示例用法
new_image_path = "D:/Program Files/pythonProject/2.jpg"  # 替换为用户上传的图片路径
top_image_paths, top_similarities = compare_image_with_database(new_image_path, feature_dir, image_dir)

# 输出相似度大于0.8的图片数量和最高相似度
if len(top_similarities) > 0 and top_similarities[0] > 0.8:
    print(f"共有{len(top_similarities)}张图片相似度大于0.8，其中相似度最高达到了{round(top_similarities[0], 3)}")
else:
    print(f"没有图片相似度超过0.8，最高的相似度为{round(top_similarities[0], 3)}")

display_images(new_image_path, top_image_paths)
