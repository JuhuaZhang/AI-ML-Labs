import warnings

# 忽视警告
warnings.filterwarnings("ignore")

import cv2
from PIL import Image
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader

# 导入已经写好的Python文件
from torch_py.Utils import plot_image
from torch_py.MTCNN.detector import FaceDetector
from torch_py.MobileNetV1 import MobileNetV1
from torch_py.FaceRec import Recognition

# 1.加载数据并进行数据处理
# 调整图片尺寸
def letterbox_image(image, size):
    """
    调整图片尺寸
    :param image: 用于训练的图片
    :param size: 需要调整到网络输入的图片尺寸
    :return: 返回经过调整的图片
    """
    new_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return new_image


# 制作训练时所需的批量数据集
def processing_data(data_path, height=224, width=224, batch_size=32, test_split=0.1):
    """
    数据处理部分
    :param data_path: 数据路径
    :param height:高度
    :param width: 宽度
    :param batch_size: 每次读取图片的数量
    :param test_split: 测试集划分比例
    :return: 
    """
    transforms = T.Compose(
        [
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.1),  # 进行随机水平翻转
            T.RandomVerticalFlip(0.1),  # 进行随机竖直翻转
            T.ToTensor(),  # 转化为张量
            T.Normalize([0], [1]),  # 归一化
        ]
    )

    dataset = ImageFolder(data_path, transform=transforms)
    # 划分数据集
    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    # 创建一个 DataLoader 对象
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_data_loader, valid_data_loader


# 绘制图片
def show_tensor_img(img_tensor):
    img = img_tensor[0].data.numpy()
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    img = np.array(img)
    plot_image(img)


# 2.如果有预训练模型，则加载预训练模型；如果没有则不需要加载
# MTCNN权重文件
pnet_path = "./torch_py/MTCNN/weights/pnet.npy"
rnet_path = "./torch_py/MTCNN/weights/rnet.npy"
onet_path = "./torch_py/MTCNN/weights/onet.npy"
data_path = "./datasets/5f680a696ec9b83bb0037081-momodel/data/image"

# 加载 MobileNet 的预训练模型权
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
train_data_loader, valid_data_loader = processing_data(
    data_path=data_path, height=160, width=160, batch_size=32
)
modify_x, modify_y = torch.ones((32, 3, 160, 160)), torch.ones((32))

epochs = 30
model = MobileNetV1(classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 优化器
print("加载完成...")

# 3.创建模型和训练模型，训练模型时尽量将模型保存在 results 文件夹
# 学习率下降的方式，acc三次不下降就下降学习率继续训练，衰减学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "max", factor=0.01, patience=2
)
# 损失函数
criterion = nn.CrossEntropyLoss()
best_loss = 1e9
best_model_weights = copy.deepcopy(model.state_dict())
loss_list = []  # 存储损失函数值
for epoch in range(epochs):
    model.train()
    for batch_idx, (x, y) in enumerate(train_data_loader, 1):
        x = x.to(device)
        y = y.to(device)
        pred_y = model(x)
        loss = criterion(pred_y, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss < best_loss:
            best_model_weights = copy.deepcopy(model.state_dict())
            best_loss = loss
    loss_list.append(best_loss)
    print(
        "step:" + str(epoch + 1) + "/" + str(epochs) + " || Total Loss: %.4f" % (loss)
    )
torch.save(model.state_dict(), "./results/temp.pth")
print("Finish Training.")
# 展示模型训练过程
plt.plot(loss_list, label="loss")
plt.legend()
plt.show()

# 4.评估模型，将自己认为最佳模型保存在 result 文件夹，其余模型备份在项目中其它文件夹，方便您加快测试通过。
# 检测图片中人数及戴口罩的人数
img = Image.open("test.jpg")
detector = FaceDetector()
recognize = Recognition(model_path="results/temp.pth")
draw, all_num, mask_nums = recognize.mask_recognize(img)
plt.imshow(draw)
plt.show()
print("all_num:", all_num, "mask_num", mask_nums)
