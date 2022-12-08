import os
import numpy as np
import jieba as jb
import jieba.analyse
import torch
from torch._C import device
import torch.nn as nn
from torch.utils import data
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, word_num, device):
        super(Net, self).__init__()
        self.Layer = nn.Sequential(
            nn.Linear(word_num, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 6),
        ).to(device)

    def forward(self, x):
        """
        前向传播
        :param x: 模型输入
        :return: 模型输出
        """
        out = self.Layer(x)
        return out


def processing_data(data_path, split_ratio=0.7):
    int2author = ["LX", "MY", "QZS", "WXB", "ZAL"]
    author_num = len(int2author)
    author2int = {author: i for i, author in enumerate(int2author)}
    # dataset = {(sentence, label), }
    dataset_init = []
    for file in os.listdir(data_path):
        if not os.path.isdir(file) and not file[0] == ".":  # 跳过隐藏文件和文件夹
            with open(
                os.path.join(data_path, file), "r", encoding="UTF-8"
            ) as f:  # 打开文件
                for line in f.readlines():
                    dataset_init.append((line, author2int[file[:-4]]))
    # 将片段组合在一起后进行词频统计
    str_full = ["" for _ in range(author_num)]
    for sentence, label in dataset_init:
        str_full[label] += sentence
    # 词频特征统计，取出各个作家前 800 的词
    words = set()
    for label, text in enumerate(str_full):
        for word in jb.analyse.extract_tags(text, topK=800, withWeight=False):
            words.add(word)

    int2word = list(words)
    word_num = len(int2word)
    word2int = {word: i for i, word in enumerate(int2word)}

    features = torch.zeros((len(dataset_init), word_num))  # 特征初始化
    labels = torch.zeros(len(dataset_init))  # 标签初始化
    for i, (sentence, author_idx) in enumerate(dataset_init):
        feature = torch.zeros(word_num, dtype=torch.float)  # 词向量初始化
        for word in jb.lcut(sentence):
            if word in words:
                feature[word2int[word]] += 1  # 构建词向量
        if feature.sum():
            feature /= feature.sum()  # 归一化
            features[i] = feature  # 加入特征集
            labels[i] = author_idx  # 加入标签集
        else:
            labels[i] = 5  # 表示识别不了作者
    dataset = data.TensorDataset(features, labels)

    # 划分数据集
    train_size = int(split_ratio * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size]
    )
    # 创建一个 DataLoader 对象
    train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = data.DataLoader(test_dataset, batch_size=1000, shuffle=True)
    return train_loader, valid_loader, word2int, int2author, word_num


def model(
    train_loader, valid_loader, save_model_path, word2int, int2author, word_num, device
):
    # 创建模型实例
    model = Net(word_num, device)
    # 查看模型参数
    for name, parameters in model.named_parameters():
        print(name, ":", parameters.size())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_acc = 0
    best_model = model.cpu().state_dict().copy()
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []

    for epoch in range(15):
        train_acc = 0
        train_loss = 0
        val_acc = 0
        val_loss = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            out = model(b_x)
            loss = loss_fn(out, b_y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accracy = np.mean((torch.argmax(out, 1) == b_y).cpu().numpy())
            # 计算每个样本的acc和loss之和
            train_acc += accracy * len(b_x)  # add
            train_loss += loss.item() * len(b_x)

            with torch.no_grad():
                for b_x, b_y in valid_loader:
                    b_x = b_x.to(device)
                    b_y = b_y.to(device)
                    out = model(b_x)
                    valid_acc = np.mean((torch.argmax(out, 1) == b_y).cpu().numpy())
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_model = model.cpu().state_dict().copy()

        with torch.no_grad():
            for b_x, b_y in valid_loader:
                b_x = b_x.to(device)
                b_y = b_y.to(device)
                out = model(b_x)
                loss = loss_fn(out, b_y.long())
                valid_acc = np.mean((torch.argmax(out, 1) == b_y).cpu().numpy())
                val_acc += valid_acc * len(b_x)  # add
                val_loss += loss.item() * len(b_x)

        train_acc /= len(train_loader.dataset)
        train_loss /= len(train_loader.dataset)
        val_acc /= len(valid_loader.dataset)
        val_loss /= len(valid_loader.dataset)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)
        print("epoch:%d | valid_acc:%.4f" % (epoch, valid_acc))
    print("best accuracy:%.4f" % (best_acc,))
    torch.save(
        {"word2int": word2int, "int2author": int2author, "model": best_model,},
        save_model_path,
    )
    # 绘制曲线
    plt.figure(figsize=(15, 5.5))
    plt.subplot(121)
    plt.plot(train_acc_list)
    plt.plot(val_acc_list)
    plt.legend(["train_acc", "val_acc"])
    plt.title("acc")
    plt.subplot(122)
    plt.plot(train_loss_list)
    plt.plot(val_loss_list)
    plt.legend(["train_loss", "val_loss"])
    plt.title("loss")
    plt.show()


def evaluate_model(valid_loader, save_model_path, device):
    config = torch.load(save_model_path)
    word2int = config["word2int"]
    int2author = config["int2author"]
    word_num = len(word2int)
    # 创建模型实例
    model = Net(word_num, device)
    model.load_state_dict(config["model"])
    int2author.append(int2author[0])
    with torch.no_grad():
        for b_x, b_y in valid_loader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            out = model(b_x)
            valid_acc = np.mean((torch.argmax(out, 1) == b_y).cpu().numpy())
    print("evaluate_model | valid_acc:%.4f" % (valid_acc))


def main():
    data_path = "./dataset/"
    save_model_path = "./results/my_model.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_val_split = 0.7  # 训练集比重
    train_loader, valid_loader, word2int, int2author, word_num = processing_data(
        data_path, split_ratio=train_val_split
    )
    model(
        train_loader,
        valid_loader,
        save_model_path,
        word2int,
        int2author,
        word_num,
        device,
    )
    evaluate_model(valid_loader, save_model_path, device)


if __name__ == "__main__":
    main()
