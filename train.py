import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 设置字体以支持中文显示
plt.rcParams['font.family'] = 'SimHei'  # 使用黑体 ('SimHei')，你也可以选择其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

def train_model():
    # 加载 YOLOv8s 模型
    model = YOLO("D:/Program Files/pythonProject/runs/segment/模型1.2/weights/best.pt")

    # 使用模型自带的优化器
    model.train(
        data="D:/Program Files/pythonProject/dataset.yaml",  # 数据集配置文件路径
        epochs=100,  # 设置总训练轮次
        imgsz=640,  # 图像尺寸

        lr0=0.004,  # 初始学习率
        lrf=0.0001,  # 最终学习率
        batch=12,  # 批次大小
        verbose=False,  # 禁用详细输出，只显示进度条
        plots=True  # 自动生成训练结果的图表
    )

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()  # 防止 RuntimeError
    train_model()
