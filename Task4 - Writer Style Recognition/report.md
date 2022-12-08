<center><h3>作家风格识别
  </h3></center>

<center>3190103683 张嘉浩
  </center>
#### 实验介绍

##### 1.1 实验背景

作家风格是作家在作品中表现出来的独特的审美风貌。通过分析作品的写作风格来识别作者这一研究有很多应用，比如可以帮助人们鉴定某些存在争议的文学作品的作者、判断文章是否剽窃他人作品等。作者识别其实就是一个文本分类的过程，文本分类就是在给定的分类体系下，根据文本的内容自动地确定文本所关联的类别。写作风格学就是通过统计的方法来分析作者的写作风格，作者的写作风格是其在语言文字表达活动中的个人言语特征，是人格在语言活动中的某种体现。

##### 1.2 实验要求

a）建立深度神经网络模型，对一段文本信息进行检测识别出该文本对应的作者。

b）绘制深度神经网络模型图、绘制并分析学习曲线。

c）用准确率等指标对模型进行评估。  

##### 1.3 实验环境

可以使用基于 Python 分词库进行文本分词处理，使用 Numpy 库进行相关数值运算，使用 Keras 等框架建立深度学习模型等。

##### 1.4 参考资料

jieba：https://github.com/fxsjy/jieba 
Numpy：https://www.numpy.org/ 
Pytorch: https://pytorch.org/docs/stable/index.html 
TorchText: https://torchtext.readthedocs.io/en/latest/



#### 实验内容

##### 2.1 介绍数据集

该数据集包含了 8438 个经典中国文学作品片段，对应文件分别以作家姓名的首字母大写命名。  
数据集中的作品片段分别取自 5 位作家的经典作品，分别是：

|序号|中文名|英文名|文本片段个数|
|--|--|--|--|
|1|鲁迅|LX| 1500 条 |
|2|莫言|MY| 2219 条 |
|3|钱钟书|QZS| 1419 条 |
|4|王小波|WXB| 1300 条 |
|5|张爱玲|ZAL| 2000 条 |

+ 其中截取的片段长度在 100~200 个中文字符不等
+ 数据集路径为 `dataset/` 以作者名字首字母缩写命名

##### 2.2 数据集预处理

在做文本挖掘的时候，首先要做的预处理就是分词。
英文单词天然有空格隔开容易按照空格分词，但是也有时候需要把多个单词做为一个分词，比如一些名词如 "New York" ，需要做为一个词看待。
而中文由于没有空格，分词就是一个需要专门去解决的问题了。
这里我们使用 jieba 包进行分词，使用**精确模式**、**全模式**和**搜索引擎模式**进行分词对比。
更多方法参考：https://github.com/fxsjy/jieba   

**使用 TF-IDF 算法统计各个作品的关键词频率**  

TF-IDF（term frequency–inverse document frequency，词频-逆向文件频率）是一种用于信息检索与文本挖掘的常用加权技术。  

* TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。           
  * 字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。  

* TF-IDF的主要思想是：如果某个单词在一篇文章中出现的频率TF高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类。  
  * 这里我们使用 jieba 中的默认语料库来进行关键词抽取，并展示每位作者前 5 个关键词


##### 2.3 采用 Pytorch 建立一个简单的深度神经网络模型

通过 Pytorch 构建深度学习模型的步骤如下：
+ 准备数据，构建Dataset
+ 定义模型、损失函数和优化器
+ 迭代训练，进行反向传播和梯度下降
+ 模型保存和测试评估



#### 作业

在按照教程搭建神经网络进行训练之后，效果并不是很理想，在查找了很多资料之后，我选择了词向量构建方式来构建自己的神经网络，代码如下。

##### 3.1 训练函数如下

```python
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
            nn.Linear(1024, 6)
        ).to(device)

    def forward(self, x):
        out = self.Layer(x)
        return out

def processing_data(data_path, split_ratio=0.7):
    int2author = ['LX', 'MY', 'QZS', 'WXB', 'ZAL']
    author_num = len(int2author)
    author2int = {author: i for i, author in enumerate(int2author)}
    # dataset = {(sentence, label), }
    dataset_init = []
    for file in os.listdir(data_path):
        if not os.path.isdir(file) and not file[0] == '.':  # 跳过隐藏文件和文件夹
            with open(os.path.join(data_path, file), 'r',  encoding='UTF-8') as f:  # 打开文件
                for line in f.readlines():
                    dataset_init.append((line, author2int[file[:-4]]))
    # 将片段组合在一起后进行词频统计
    str_full = ['' for _ in range(author_num)]
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

    features = torch.zeros((len(dataset_init), word_num)) # 特征初始化
    labels = torch.zeros(len(dataset_init)) # 标签初始化
    for i, (sentence, author_idx) in enumerate(dataset_init):
        feature = torch.zeros(word_num, dtype=torch.float) # 词向量初始化
        for word in jb.lcut(sentence):
            if word in words:
                feature[word2int[word]] += 1 # 构建词向量
        if feature.sum():
            feature /= feature.sum() # 归一化
            features[i] = feature # 加入特征集
            labels[i] = author_idx # 加入标签集
        else:
            labels[i] = 5  # 表示识别不了作者
    dataset = data.TensorDataset(features, labels)

    # 划分数据集
    train_size = int(split_ratio * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size])
    # 创建一个 DataLoader 对象
    train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = data.DataLoader(test_dataset, batch_size=1000, shuffle=True)
    return train_loader, valid_loader, word2int, int2author, word_num

def model(train_loader, valid_loader, save_model_path, word2int, int2author, word_num, device):
    # 创建模型实例
    model = Net(word_num, device)
    # 查看模型参数
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
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
            train_acc += accracy*len(b_x)  # add
            train_loss += loss.item()*len(b_x)

            with torch.no_grad():
                for b_x, b_y in valid_loader:
                    b_x = b_x.to(device)
                    b_y = b_y.to(device)
                    out = model(b_x)
                    valid_acc = np.mean(
                        (torch.argmax(out, 1) == b_y).cpu().numpy())
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_model = model.cpu().state_dict().copy()

        with torch.no_grad():
            for b_x, b_y in valid_loader:
                b_x = b_x.to(device)
                b_y = b_y.to(device)
                out = model(b_x)
                loss = loss_fn(out, b_y.long())
                valid_acc = np.mean(
                    (torch.argmax(out, 1) == b_y).cpu().numpy())
                val_acc += valid_acc*len(b_x)  # add
                val_loss += loss.item()*len(b_x)

        train_acc /= len(train_loader.dataset)
        train_loss /= len(train_loader.dataset)
        val_acc /= len(valid_loader.dataset)
        val_loss /= len(valid_loader.dataset)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)
        print('epoch:%d | valid_acc:%.4f' % (epoch, valid_acc))
    print('best accuracy:%.4f' % (best_acc, ))
    torch.save({
        'word2int': word2int,
        'int2author': int2author,
        'model': best_model,
    }, save_model_path)

def evaluate_model(valid_loader, save_model_path, device):
    config = torch.load(save_model_path)
    word2int = config['word2int']
    int2author = config['int2author']
    word_num = len(word2int)
    # 创建模型实例
    model = Net(word_num, device)
    model.load_state_dict(config['model'])
    int2author.append(int2author[0])
    with torch.no_grad():
        for b_x, b_y in valid_loader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            out = model(b_x)
            valid_acc = np.mean((torch.argmax(out, 1) == b_y).cpu().numpy())
    print('evaluate_model | valid_acc:%.4f' % (valid_acc))

def main():
    data_path = './dataset/'
    save_model_path = './results/my_model.pth'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_val_split = 0.7  # 训练集比重
    train_loader, valid_loader, word2int, int2author, word_num = processing_data(
        data_path, split_ratio=train_val_split)
    model(train_loader, valid_loader, save_model_path,
          word2int, int2author, word_num, device)
    evaluate_model(valid_loader, save_model_path, device)

if __name__ == '__main__':
    main()
```
##### 3.2 预测函数如下

```python
import torch
import torch.nn as nn
from train_main import Net
import jieba as jb

test_model_path = 'results/my_model.pth'
test_model = torch.load(test_model_path)

word2int = test_model['word2int']
int2author = test_model['int2author']
word_num = len(word2int)

device = torch.device('cpu')
# 创建模型实例
model = Net(word_num, device)
model.load_state_dict(test_model['model'])
int2author.append(int2author[0])

def predict(text):
    """
    :param text: 中文字符串
    :return: 字符串格式的作者名缩写
    """
    feature = torch.zeros(word_num)
    for word in jb.lcut(text):
        if word in word2int:
            feature[word2int[word]] += 1
    feature /= feature.sum()
    model.eval()
    out = model(feature.unsqueeze(dim=0))
    pred = torch.argmax(out, 1)[0]
    return int2author[pred]

if __name__ == '__main__':
    target_text = "中国中流的家庭，教孩子大抵只有两种法。其一是任其跋扈，一点也不管，骂人固可，打人亦无不可，在门内或门前是暴主，是霸王，但到外面便如失了网的蜘蛛一般，立刻毫无能力。其二，是终日给以冷遇或呵斥，甚于打扑，使他畏葸退缩，彷佛一个奴才，一个傀儡，然而父母却美其名曰“听话”，自以为是教育的成功，待到他们外面来，则如暂出樊笼的小禽，他决不会飞鸣，也不会跳跃。"
    print(predict(target_text))
```


#### 训练结果

##### 4.1 训练结果

测试准确率为48/50，表现良好。

##### 4.2 体会心得

本次作业我调参了很久，但效果都不好，最终在同学的帮助下选择了词向量构建方式搭建神经网络，最终效果非常良好。