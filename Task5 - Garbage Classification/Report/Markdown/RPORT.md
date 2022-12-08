<center><h1>
  基于神经网络的垃圾分类
  </h1></center>

<center>
  朱博医 代洋飞 张嘉浩
  </center>
**摘要：**随着社会的飞速发展，生活中产生的消耗废品日益剧增，如何更好地分类与回收这些“垃圾”已经成为了急需解决的问题。为了我国能够更好更快的建立健全城市垃圾分类处理制度以及方便人们对于垃圾分类有更全面的认识，本文利用人工智能技术，设计了基于深度学习的垃圾分类模型，对包含2700多张图片的数据集进行六分类训练。训练结果表明，在全连接神经网络，卷积神经网络和MobileNetV2神经网络中，MoblieNetV2网络的效果最好，准确率可以到达91.66%，实现了较好的分类效果。

**关键词：**垃圾分类; 卷积神经网络; MobileNetV2



<center><h1>
   Garbage Classification On Neural Network
   </h1></center>


<center>
   Zhu Boyi, Dai Yangfei, Zhang Jiahao
   </center>
**Summary:** With the rapid development of society, the waste consumption generated in life is increasing rapidly. How to better classify and recycle these "garbages" has become an urgent problem to be solved. In order to establish and improve the urban garbage classification and treatment system better and faster in China and facilitate people to have a more comprehensive understanding of garbage classification, this paper uses artificial intelligence technology to design a garbage classification model based on deep learning and conduct six classification training for data sets containing more than 2700 pictures. The training results show that among the fully connected neural networks, convolutional neural networks and MobileNetV2 neural networks, the MoblieNetV2 network has the best effect, with an accuracy of 91.66%, achieving a good classification effect.

**Key Words:** garbage sorting; Convolutional neural network; MobileNetV2



## 1 引言

### 1.1 选题背景

垃圾分类，一般是指按一定规定或标准将垃圾分类储存、分类投放和分类搬运，从而转变成公共资源的一系列活动的总称。自今年7月1日起，上海市将正式实施《上海市生活垃圾管理条例》。垃圾分类，看似是微不足道的“小事”，实则关系到13亿多人生活环境的改善，理应大力提倡。

### 1.2 研究意义

分类的目的是提高垃圾的资源价值和经济价值，力争物尽其用。进行垃圾分类收集可以减少垃圾处理量和处理设备，降低处理成本，减少土地资源的消耗，具有社会、经济、生态等几方面的效益。生活垃圾由于种类繁多，具体分类缺乏统一标准，大多人在实际操作时会“选择困难”，基于深度学习技术建立准确的分类模型，利用技术手段改善人居环境。

### 1.3 相关文献综述

早期, 学者们只能借助经典的图像分类算法完成垃圾图像分类任务, 这要通过手动提取的图像特征并结合相应的分类器完成. 吴健等利用颜色和纹理特征, 初步完成了废物垃圾识别. 由于不同数据集的图像背景、尺寸、质量不尽相同, 传统算法需要根据相应数据人工提取不同的特征, 算法的鲁棒性较差, 并且处理方式复杂, 所需时间较长, 无法达到实时的效果. 随着卷积神经网络(Convolution Neural Network, CNN)的飞速发展, 深度学习广泛应用于图像识别领域. 作为数据驱动的算法, CNN具有强大的特征拟合能力, 可以有效、自动地提取图像特征, 并具有较快的运行速度. 2012年, AlexNet取得了ImageNet图像分类竞赛的冠军, 标志着深度学习的崛起. 随后几年, GoogleNet、VGGNet、ResNet等算法提升了图像分类的精度, 并成功应用于人脸识别、车辆检测等多个领域. 垃圾图像分类, 在深度学习算法的帮助下同样取得了较大的突破. 斯坦福大学的Yang等建立了TrashNet Dataset公开数据集, 包含6个类别, 共计2527张图片. Ozkaya等通过对比不同CNN网络的分类能力, 搭建神经网络(本文称之为TrashNet)并进行参数微调, 在数据集TrashNet Dataset上取得了97.86%的准确率, 是目前这一数据集上最佳分类网络. 在非公开数据集方面, Mittal等自制了2561张的垃圾图片数据集GINI, 使用GarbNet模型, 得到了87.69%的准确率. 国内方面, 郑海龙等用SVM方法进行了建筑垃圾分类方面的研究. 向伟等使用分类网络CaffeNet, 调整卷积核尺寸和网络深度, 使其适用于水面垃圾分类, 在其自制的1500张图片数据集上取得了95.75%的识别率. 2019年, 华为举办垃圾图像分类竞赛, 构建了样本容量为一万余张的数据集, 进一步促进了该领域的发展。



## 2 算法描述与求解结果

### 2.1 问题描述

小组需要以 MobileNetV2+ 垃圾分类数据集为例，使用深度学习框架（如Pytorch/Tensorflow）在CPU/GPU 平台上进行训练，实现对26种垃圾进行分类。或者自己搭建神经网络，实现垃圾分类。

### 2.2 训练样本

训练样本由 `mo平台` 提供，图片分为干垃圾，可回收物，湿垃圾和有害垃圾四个大类共26种物品共2375张图片组成，在训练过程中使用以下函数将训练样本划分为训练集和测试集两大类，供后续的训练和验证所用：

```python
    train_db, val_db = torch.utils.data.random_split(dataset,[2250,125])
```



### 2.3 搭建过程

对于垃圾分类问题，我们小组主要进行了三轮模型的调试，具体算法如下：

#### 2.3.1 第一轮尝试：DNN

我们首先搭建了简单的一个三层全连接神经网络（DNN），我们首先将输入的图片展开为一个一维向量：

```python
    # 将输入展平
    inputs = Input(shape=input_shape)
    dnn = Flatten()(inputs)
```

然后将一维向量输入到搭建完的神经网络中，每一层网络由BN层，激活层组成，第一层网络的结构如下：

```python
    # Dense 全连接层
    dnn = Dense(6)(dnn)
    dnn = BatchNormalization(axis=-1)(dnn)
    dnn = Activation('sigmoid')(dnn)
    dnn = Dropout(0.25)(dnn)
```

在三层网络中，我们分别采用了`sigmoid`, `relu` 和 `softmax` 激活函数。通过对该神经网络进行长达一个小时的训练，但是由于图像信息很大，而全连接层并不适合，测试集准确度很低，最高仅达到0.3左右。



#### 2.3.2 第二轮尝试：CNN

卷积神经网络通常被用于处理多阵列形式的数据，例如有三个二维阵列组成的RGB图像，每个二位整列包含的是三个彩色通道的像素强度。该类网络的背后有四个关键思想，它们利用了自然信号的特性，分别是：本地连接、共享权重、池化和多层的使用。 

由于CNN的特征检测层通过训练数据进行学习，所以在使用CNN时，避免了显式的特征抽取，而隐式地从训练数据中进行学习；再者由于同一特征映射面上的神经元权值相同，所以网络可以并行学习，这也是卷积网络相对于神经元彼此相连网络的一大优势。

##### （1）网络结构

**l 数据输入层/ Input layer**

**l 卷积计算层/ CONV layer**

**l ReLU激励层 / ReLU layer**

**l 池化层 / Pooling layer**

**l 全连接层 / FC layer**

​	数据输入层对图像进行去均值、归一化等预处理；卷积、池化的过程将一张图片的维度进行了压缩，有特征降维和防止过拟合等作用；全连接层通常在卷积神经网络尾部负责网络与输出的联接。从图示上我们不难看出卷积网络的精髓就是适合处理结构化数据，而该数据在跨区域上依然有关联。

我们自己搭建了一个五层的卷积神经网络（CNN），每一层神经网络由卷积层，BN（归一化）层，激活层与最大池化层组成，以第一层结构为例：

```python
# conv1: Conv2d -> BN -> ReLU -> MaxPool
self.conv1 = nn.Sequential(
     nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
     nn.BatchNorm2d(16),
     nn.ReLU(), 
     nn.MaxPool2d(kernel_size=2, stride=1),
     )
```



##### （2）图片变换

我们对图片进行了一系列变换，如旋转、翻转、灰度化以增强稳定度；

```python
transform = transforms.Compose([
    transforms.Resize((128,128)), 		 #尺寸变换
    transforms.RandomRotation((30,30)),  #旋转
    transforms.RandomVerticalFlip(0.1),  #垂直翻转
    transforms.RandomGrayscale(0.1),  	 #灰度化
    transforms.ToTensor(),				 #转化张量
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
```



##### （3）调整超参数

在训练过程中主要调整以下参数：

```python
# #hyper parameter
batch_size = 32
num_epochs = 50
lr = 0.0001
num_classes = 25
image_size = 128 #### 64
```



##### （4）训练过程

对于该神经网络，设置 `epoch = 50` 进行训练，得到训练的损失曲线如下图所示：

<img src=".\2.png" style="zoom:67%;" />

<center>
  图1：卷积神经网络训练结果
</center>

虽然训练集的损失值随着epoch的增加而接近于0，但是神经网络在测试集的表现程度并不好，从图2可以看到，随着训练次数的增加，正确率在0.65左右波动，最后正确率在70%左右，说明模型进入了过拟合阶段。

<img src=".\4.png" style="zoom:67%;" />

<center>
  图2：卷积神经网络在测试集的结果
</center>


<img src=".\3.png" style="zoom:67%;" />

<center>
  图3：卷积神经网络在验证集的结果
</center>

它虽然在训练集的表现中越来越好，但是由于训练集样本数量过少，使得训练出的网络不能很好的应用于所有的情况。

#### 2.3.3 第三轮尝试：MobileNetV2

此次尝试采用了  `mo平台` 提供的利用 MindSpore 搭建 MobileNetV2网络模型的基础上进行参数修改。

由于在深度学习计算中，从头开始训练一个实用的模型通常非常耗时，需要大量计算能力。常用的数据如 OpenImage、ImageNet、VOC、COCO 等公开大型数据集，规模达到几十万甚至超过上百万张。网络和开源社区上通常会提供这些数据集上预训练好的模型。大部分细分领域任务在训练网络模型时，如果不使用预训练模型而从头开始训练网络，不仅耗时，且模型容易陷入局部极小值和过拟合。因此大部分任务都会选择预训练模型，在其上做微调（也称为 Fine-Tune）。



##### （1）网络简介

###### MobileNet

在现实场景下，诸如移动设备、嵌入式设备、自动驾驶等等，计算能力会受到限制，MobileNet由此提出。

相较于传统网络，它

1. 提出了MobileNet架构，使用深度可分离卷积（depthwise separable convolutions）替代传统卷积。
2. 在MobileNet网络中还引入了两个收缩超参数（shrinking hyperparameters）：宽度乘子（width multiplier）和分辨率乘子（resolution multiplier）。



###### MobileNetV2

V2 主要引入了两个改动：Linear Bottleneck和 Inverted Residual Blocks。

1) Inverted Residual BlocksMobileNetV2 结构基于 inverted residual。其本质是一个残差网络设计，传统 Residual block 是 block 的两端 channel 通道数多，中间少，而 MobileNetV2 设计的 inverted residual 是 block 的两端 channel 通道数少，block 内 channel 多，类似于沙漏和梭子形态的区别。另外保留 Depthwise Separable Convolutions。

2) Linear Bottlenecks感兴趣区域在 ReLU 之后保持非零，近似认为是线性变换。ReLU 能够保持输入信息的完整性，但仅限于输入特征位于输入空间的低维子空间中。对于低纬度空间处理，论文中把 ReLU 近似为线性转换。



##### （2）数据准备

将脚本、预训练模型的 Checkpoint 和数据集组织为如下形式：

```bash
├── main.ipynb # 入口Jupyter Notebook文件
│
├── src_mindspore
│   ├── dataset.py
│   ├── mobilenetv2.py
│   └── mobilenetv2-200_1067_gpu_cpu.ckpt
│
├── results/mobilenetv2.mindir # 待生成的MindSpore0.5.0模型文件
│
├── train_main.py # 将 main.ipynb Notebook 训练模型代码转化为py文件
│
└── datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100/ # 数据集
    ├── train/
    ├── val/
    └── label.txt
```



##### （3）参数配置

配置后续训练、验证、推理用到的参数。可以调整以下超参以提高模型训练后的验证精度：

- `epochs`：在训练集上训练的代数；
- `lr_max`：学习率，或者动态学习率的最大值；
- `decay_type`：学习率下降策略；
- `momentum`：Momentum优化器的动量参数，通常为0.9；
- `weight_decay`：正则化项的系数。

```python
# 训练超参
config = EasyDict({
    "num_classes": 26, # 分类数，即输出层的维度
    "reduction": 'mean', # mean, max, Head部分池化采用的方式
    "image_height": 224,
    "image_width": 224,
    "batch_size": 24, # 鉴于CPU容器性能，太大可能会导致训练卡住
    "eval_batch_size": 10,
    "epochs": 4, # 请尝试修改以提升精度
    "lr_max": 0.01, # 请尝试修改以提升精度
    "decay_type": 'constant', # 请尝试修改以提升精度
    "momentum": 0.8, # 请尝试修改以提升精度
    "weight_decay": 3.0, # 请尝试修改以提升精度
    "dataset_path": "./datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100",
    "features_path": "./results/garbage_26x100_features", # 临时目录，保存冻结层Feature Map，可随时删除
    "class_index": index,
    "save_ckpt_epochs": 1,
    "save_ckpt_path": './results/ckpt_mobilenetv2',
    "pretrained_ckpt": './src_mindspore/mobilenetv2-200_1067_cpu_gpu.ckpt',
    "export_path": './results/mobilenetv2.mindir'  
})
```



##### （4）训练策略

一般情况下，模型训练时采用静态学习率，如0.01。随着训练步数的增加，模型逐渐趋于收敛，对权重参数的更新幅度应该逐渐降低，以减小模型训练后期的抖动。所以，模型训练时可以采用动态下降的学习率，常见的学习率下降策略有：

- polynomial decay/square decay;
- cosine decay;
- exponential decay;
- stage decay.

这里实现 cosine decay 和 square decay 下降策略。

```python
def build_lr(total_steps, lr_init=0.0, lr_end=0.0, lr_max=0.1, warmup_steps=0, decay_type='cosine'):
    lr_init, lr_end, lr_max = float(lr_init), float(lr_end), float(lr_max)
    decay_steps = total_steps - warmup_steps
    lr_all_steps = []
    inc_per_step = (lr_max - lr_init) / warmup_steps if warmup_steps else 0
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + inc_per_step * (i + 1)
        else:
            if decay_type == 'cosine':
                cosine_decay = 0.5 * (1 + math.cos(math.pi * (i - warmup_steps) / decay_steps))
                lr = (lr_max - lr_end) * cosine_decay + lr_end
            elif decay_type == 'square':
                frac = 1.0 - float(i - warmup_steps) / (total_steps - warmup_steps)
                lr = (lr_max - lr_end) * (frac * frac) + lr_end
            else:
                lr = lr_max
        lr_all_steps.append(lr)

    return lr_all_steps
```



##### （5）模型训练

在模型训练过程中，通过添加检查点（Checkpoint）用于保存模型的参数，以便进行推理及中断后再训练使用。使用场景如下：

- 训练后推理场景
    - 模型训练完毕后保存模型的参数，用于推理或预测操作。
    - 训练过程中，通过实时验证精度，把精度最高的模型参数保存下来，用于预测操作。
- 再训练场景
    - 进行长时间训练任务时，保存训练过程中的Checkpoint文件，防止任务异常退出后从初始状态开始训练。
    - Fine-tuning（微调）场景，即训练一个模型并保存参数，基于该模型，面向第二个类似任务进行模型训练。

这里加载ImageNet数据上预训练的MobileNetv2进行Fine-tuning，并在训练过程中保存Checkpoint。训练有两种方式：
- 方式一：冻结网络的Backbone，只训练修改的FC层（Head）。其中，Backbone再全量数据集上做一遍推理，得到Feature Map，将Feature Map作为训练Head的数据集，可以极大节省训练时间。
- 方式二：先冻结网络的Backbone，只训练网络Head；再对Backbone+Head做整网做微调。

###### 提取特征集

将冻结层在全量训练集上做一遍推理，然后保存FeatureMap，作为修改层的数据集。

```python
def extract_features(net, dataset_path, config):
    if not os.path.exists(config.features_path):
        os.makedirs(config.features_path)
    dataset = create_dataset(config=config)
    step_size = dataset.get_dataset_size()
    if step_size == 0:
        raise ValueError("The step_size of dataset is zero.")

    data_iter = dataset.create_dict_iterator()
    for i, data in enumerate(data_iter):
        features_path = os.path.join(config.features_path, f"feature_{i}.npy")
        label_path = os.path.join(config.features_path, f"label_{i}.npy")
        if not os.path.exists(features_path) or not os.path.exists(label_path):
            image = data["image"]
            label = data["label"]
            features = net(image)
            np.save(features_path, features.asnumpy())
            np.save(label_path, label.asnumpy())
        print(f"Complete the batch {i+1}/{step_size}")
    return

backbone = MobileNetV2Backbone()
load_checkpoint(config.pretrained_ckpt, net=backbone)
extract_features(backbone, config.dataset_path, config)
```



##### （6）参数调整

我们对网络的调整主要是以下几个参数：

###### "reduction"，部分池化方式

池化层的目的就是对大量的特征信息进行过滤，去除其中的冗余信息并筛选出最具代表性的特征信息，因此可以把池化层当作是一个滤波器。池化层的作用包括减少网络中参数的数量、压缩数据以及减少网络的过拟合。池化层里面主要包含了两个参数，分别是步长和池化核大小。池化核以滑动窗口的方式对输入的特征图进行处理，经过不同的池化函数的计算，得到相应的关键特征，其中每个池化层中的池化函数是固定的，一般不需要再引入其他参数。池化函数是池化层的核心，池化函数的不同也就对应着不同的池化方法。一个较好的池化方法通常能够在删除大量的无关信息的同时并且尽可能多的保留关键信息，进而在很大程度上提升整个卷积神经网络的性能。

池化方法中最常见的方法是最大池化和平均池化。最大池化只保留池化框中的最大值，因而最大池化可以有效提取出特征图中最具代表性的信息。平均池化则计算出池化框中所有值的均值，因而可以平均获取特征图中的所有信息，进而不致丢失过多关键信息。这两种方法由于计算简单且效果较好因而被广泛利用在了各种结构的卷积神经网络中，但这两种方法的缺点也是不可忽视的。最大池化由于完全删除了最大值以外的其他值，这往往导致保留了特征图中的前景信息而忽略了所有的背景信息；而平均池化由于取得了所有值之和的均值，虽然对特征图中的背景信息有所保留，但是无法将特征图中的前景信息和背景信息有效地区分开。

基于对两种池化方法（ `max` 和 `mean`）的考虑，在该网络中我们选择了最大池化方法，那么在采取了最大池化之后，整个网络的准确率有了很大的提高，从开始的 60% 一下子上升到 80%。



###### “batch_size”，批尺寸

Batch 决定了梯度下降的方向。如果数据集比较小，完全可以采用全数据集 （ Full Batch Learning ）的形式，这样做至少有 2 个好处：其一，由全数据集确定的方向能够更好地代表样本总体，从而更准确地朝向极值所在的方向。其二，由于不同权重的梯度值差别巨大，因此选取一个全局的学习率很困难。 Full Batch Learning 可以使用Rprop (弹性反向传播)只基于梯度符号并且针对性单独更新各权值。对于更大的数据集，以上 2 个好处又变成了 2 个坏处：其一，随着数据集的海量增长和内存限制，一次性载入所有的数据进来变得越来越不可行。其二，以 Rprop 的方式迭代，会由于各个 Batch 之间的采样差异性，各次梯度修正值相互抵消，无法修正。

在合理范围内，增大 Batch_Size 可以提高内存利用率。跑完一次 epoch（全数据集）所需的迭代次数减少，对于相同数据量的处理速度进一步加快。在一定范围内，一般来说 Batch_Size 越大，其确定的下降方向越准，引起训练震荡越小。但 Batch_Size 过大，可能使得内存容量不足而训练失败，同时跑完一次 epoch（全数据集）所需的迭代次数减少，要想达到相同的精度，其所花费的时间大大增加了，从而对参数的修正也就显得更加缓慢。

我们根据尝试，选择了最佳的 `batch_size = 26`, 使得训练性能达到最佳。



###### “lr_rate”，最大学习率

学习率(Learning rate)作为监督学习以及深度学习中重要的超参，其决定着目标函数能否收敛到局部最小值以及何时收敛到最小值。合适的学习率能够使目标函数在合适的时间内收敛到局部最小值。当学习率设置的过小时，收敛过程将变得十分缓慢。而当学习率设置的过大时，梯度可能会在最小值附近来回震荡，甚至可能无法收敛。

同时，学习率和batchsize的紧密相连，通常当我们增加batchsize为原来的N倍时，要保证经过同样的样本后更新的权重相等，按照线性缩放规则，学习率应该增加为原来的N倍。但是如果要保证权重的方差不变，则学习率应该增加为原来的sqrt(N)倍。

那么经过大量尝试，我们将学习率最终确定为了 `lr_max = 0.003` 以达到相对来说的最优值。



##### （7）训练结果

将参数调整为合适值进行训练，训练35轮，最终得到准确率为 87.7%，损失曲线如下：

<img src=".\1.png" style="zoom:88%;" />

<center>
  图3：MobileNetV2训练结果
</center>
**（8）预测结果**



<img src=".\5.png" style="zoom:100%;" />

<center>
  图4：预测帽子效果
</center>
<img src=".\6.png" style="zoom:100%;" />

<center>
  图5：预测报纸效果
</center>

在所给的验证集中，大多数种类垃圾可以全部识别正确，个别垃圾种类会识别错误一张。



## 3 比较分析\&结论

在此次实验中我们实验的网络主要是CNN和MobileNet。

首先，我们之所以选择CNN是因为卷积网络的精髓就是适合处理结构化数据，而该数据在跨区域上依然有关联，常被用于图像分析。它具有参数共享机制的优点，在卷积层中每个神经元连接数据窗的权重是固定的，每个神经元只关注一个特性。神经元就是图像处理中的滤波器，比如边缘检测专用的Sobel滤波器，即卷积层的每个滤波器都会有自己所关注一个图像特征，比如垂直边缘，水平边缘，颜色，纹理等等，这些所有神经元加起来就好比就是整张图像的特征提取器集合。另外，CNN也无需手动选取特征，训练好权重，即得输入特征（自动的），分类效果好一般比较好。但是，CNN的缺点就是需要调参，需要大样本量，训练最好要GPU，而在本次实验中，我们认为我们自己搭建的CNN的准确率不太理想的原因是因为网络模型还不是最优，并且样本数量不足。

而在MobileNet网络中，虽然最后准确率可以稳定在85%以上，并且预测效果也算比较理想，在所给的验证集中，大多数种类垃圾可以全部识别正确，个别垃圾种类会识别错误一张。但是我们认为MobileNet的可调度很小，我们也不能很清楚的了解到他其中的具体网络结构。



## 4 研究感想与组员分工

##### 研究感想 

在搭建神经网络的过程中，我们从结构上对神经网络有了新的理解。搭建一个网络与学习一个网络有很大的不同，需要注意的地方也更多，比如网络的大小和深度等等。那么在调整参数的过程中呢，我们也意识到调参并不是盲目的，需要结合各个参数的意义和网络的输出结果进行针对性的调整。在搜集资料的过程中，我们小组也通过前人的文章对人工智能与机器学习这个领域有了更加深入的了解和全新的认识，收获颇丰。



##### 成员分工

张嘉浩、代洋飞：搭建神经网络并调参、报告撰写

朱博医：数据集读取、PPT制作、调参、报告撰写




## 5 参考文献

| [1]  | 吕思敏. 以史为鉴, 开启垃圾分类新时代. [城乡建设, 2020(3): 30-32.](http://d.old.wanfangdata.com.cn/Periodical/cxjs202003012) |
| ---- | ------------------------------------------------------------ |
| [2]  | Abeywickrama T, Cheema MA, Taniar D. K-nearest neighbors on road networks: A journey in experimentation and in-memory implementation. Proceedings of the VLDB Endowment, 2016, 9(6): 492-503. [DOI:10.14778/2904121.2904125](http://dx.doi.org/10.14778/2904121.2904125) |
| [3]  | Lowe DG. Distinctive image features from scale-invariant keypoints. International Journal of Computer Vision, 2004, 60(2): 91-110. [DOI:10.1023/B:VISI.0000029664.99615.94](http://dx.doi.org/10.1023/B:VISI.0000029664.99615.94) |
| [4]  | Harri C, Stephens M. A combined corner and edge detector. Proceedings of the 4th Alvey Vision Conference. Manchester, UK. 1988. 207–217. |
| [5]  | Vapnik V. Statistical Learning Theory. New York: Wiley, 1998. 401–492. |
| [6]  | 吴健, 陈豪, 方武. 基于计算机视觉的废物垃圾分析与识别研究. 信息技术与信息化, 2016(10): 81-83. [DOI:10.3969/j.issn.1672-9528.2016.10.020](http://dx.doi.org/10.3969/j.issn.1672-9528.2016.10.020) |
| [7]  | Krizhevsky A, Sutskever I, Hinton GE. ImageNet classification with deep convolutional neural networks. Advances in Neural Information Processing Systems. Lake Tahoe, NV, USA. 2012. 1106–1114. |
| [8]  | Szegedy C, Liu W, Jia YQ, et al. Going deeper with convolutions. arXiv: 1409.4842, 2014. |
| [9]  | Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition. arXiv: 1409.1556, 2015. |
| [10] | He KM, Zhang XY, Ren SQ, et al. Deep residual learning for image recognition. Proceedings of 2016 IEEE Conference on Computer Vision and Pattern Recognition. Las Vegas, NV, USA. 2016. 770–778. |
| [11] | Ozkaya U, Seyfi L. Fine-tuning models comparisons on garbage classification for recyclability. arXiv: 1908.04393, 2019. |
| [12] | Mittal G, Yagnik KB, Garg M, et al. SpotGarbage: Smartphone app to detect garbage using deep learning. Proceedings of 2016 ACM International Joint Conference. Heidelberg, Germany. 2016. 940–945. |
| [13] | 郑龙海, 袁祖强, 殷晨波, 等. 基于机器视觉的建筑垃圾自动分类系统研究. 机械工程与自动化, 2019(6): 16-18. [DOI:10.3969/j.issn.1672-6413.2019.06.006](http://dx.doi.org/10.3969/j.issn.1672-6413.2019.06.006) |
| [14] | 向伟, 史晋芳, 刘桂华, 等. 改进CaffeNet模型在水面垃圾识别中的应用. [传感器与微系统, 2019, 38(8): 150-152, 156.](http://d.old.wanfangdata.com.cn/Periodical/cgqjs201908042) |
| [15] | Zhang XK, Wang Y, Gou MR, et al. Efficient temporal sequence comparison and classification using gram matrix embeddings on a riemannian manifold. Proceedings of 2016 IEEE Conference on Computer Vision and Pattern Recognition. Las Vegas, NV, USA. 2016. 4498–4507. |
| [16] | M, Q, Yan SC. Network in network. arXiv: 1312.4400, 2014.    |
| [17] | Kingma DP, Ba J. Adam: A method for stochastic optimization. arXiv: 1412.6980, 2017. |



## 附录：实验代码

##### DNN

```python
# Temp 1

    inputs = Input(shape=input_shape)

    # 将输入展平
    dnn = Flatten()(inputs)

    # Dense 全连接层
    dnn = Dense(6)(dnn)
    dnn = BatchNormalization(axis=-1)(dnn)
    dnn = Activation('sigmoid')(dnn)
    dnn = Dropout(0.25)(dnn)

    dnn = Dense(12)(dnn)
    dnn = BatchNormalization(axis=-1)(dnn)
    dnn = Activation('relu')(dnn)
    dnn = Dropout(0.5)(dnn)

    dnn = Dense(6)(dnn)
    dnn = BatchNormalization(axis=-1)(dnn)
    dnn = Activation('softmax')(dnn)

    outputs = dnn

    # 生成一个函数型模型
    model = Model(inputs=inputs, outputs=outputs)

```

##### CNN

```python
# Temp 2
### 浙江大学人工智能课程——CNN实现垃圾分类
### 代洋飞 张嘉浩 朱博医
### 2021/12/15

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import time
import os
from torchvision import models
import matplotlib.pyplot as plt
from torchviz import make_dot


class MyCNN(nn.Module):
    """
    网络模型
    """
    def __init__(self, image_size, num_classes):
        super(MyCNN, self).__init__()
        # conv1: Conv2d -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding='same'),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(), 
            nn.AvgPool2d(kernel_size=2, stride=1),
        )
        # conv2: Conv2d -> BN -> ReLU -> MaxPool  nn.AdaptiveAvgPool2d
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1),
        )
        # fully connected layer
        self.dp1 = nn.Dropout(0.50)
        self.fc1 = nn.Linear(12544, 2048)
        self.dp2 = nn.Dropout(0.50)
        self.fc2 = nn.Linear(2048, 256)
        self.dp3 = nn.Dropout(0.20)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        input: N * 3 * image_size * image_size
        output: N * num_classes
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # view(x.size(0), -1): change tensor size from (N ,H , W) to (N, H*W)
        x = x.view(x.size(0), -1)
        x = self.dp1(x)
        x = self.fc1(x)
        x = self.dp2(x)
        x = self.fc2(x)
        x = self.dp3(x)
        output = self.fc3(x)
        return output


def getRsn():
    model = models.resnet18(pretrained=True)
    num_fc_in = model.fc.in_features
    model.fc = nn.Linear(num_fc_in, 6)
    return model

def getMbnet():
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1280,out_features=64),
        nn.Dropout(p=0.5,inplace=False),
        nn.Linear(in_features=64,out_features=6,bias=True),
    )
    return model

def train(model, train_loader, loss_func, optimizer, device):
    """
    训练模型
    train model using loss_fn and optimizer in an epoch.
    model: CNN networks
    train_loader: a Dataloader object with training data
    loss_func: loss function
    device: train on cpu or gpu device
    """
    total_loss = 0
    # train the model using minibatch
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        # forward
        outputs = model(images)
        _,preds = torch.max(outputs.data,1)
        loss = loss_func(outputs, targets)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 100 == 0:
            print ("Step [{}/{}] Train Loss: {:.4f} Train acc".format(i+1, len(train_loader), loss.item()))
    save_model(model, save_path="results/cnn.pth")
    return total_loss / len(train_loader)


def evaluate(model, val_loader, device, name):
    """
    评估模型
    model: CNN networks
    val_loader: a Dataloader object with validation data
    device: evaluate on cpu or gpu device
    return classification accuracy of the model on val dataset
    """
    # evaluate the model
    model.eval()
    # context-manager that disabled gradient computation
    with torch.no_grad():
        correct = 0
        total = 0
        
        for i, (images, targets) in enumerate(val_loader):
            # device: cpu or gpu
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)

            # return the maximum value of each row of the input tensor in the 
            # given dimension dim, the second return vale is the index location
            # of each maxium value found(argmax)
            _, predicted = torch.max(outputs.data, dim=1)
            correct += (predicted == targets).sum().item()

            total += targets.size(0)

        accuracy = correct / total
        print('Accuracy on '+name+' Set: {:.4f} %'.format(100 * accuracy))
        return accuracy


def save_model(model, save_path="results/cnn.pth"):
    '''保存模型'''
    # save model
    torch.save(model.state_dict(), save_path)


def show_curve(ys, title):
    """
    plot curlve for Loss and Accuacy
    Args:
        ys: loss or acc list
        title: loss or accuracy
    """
    x = np.array(range(len(ys)))
    y = np.array(ys)
    plt.plot(x, y, c='b')
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.show()





def fit(model, num_epochs, optimizer, device):
    """
    train and evaluate an classifier num_epochs times.
    We use optimizer and cross entropy loss to train the model. 
    Args: 
        model: CNN network
        num_epochs: the number of training epochs
        optimizer: optimize the loss function
    """

    # loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    model.to(device)
    loss_func.to(device)

    # log train loss and test accuracy
    losses = []
    accs = []
    accst = []
    i = 0
    for epoch in range(num_epochs):

        print('Epoch {}/{}:'.format(epoch + 1, num_epochs))
        # train step
        loss = train(model, train_loader, loss_func, optimizer, device)
        losses.append(loss)

        # evaluate step
        accuracy = evaluate(model, test_loader, device, 'test')
        accuracy1 = evaluate(model, train_loader, device, 'train')
        accs.append(accuracy)
        accst.append(accuracy1)


    # show curve
    show_curve(losses, "train loss")
    show_curve(accs, "test accuracy")
    show_curve(accst, "train accuracy")

# model = models.vgg16_bn(pretrained=True)

# model_ft= models.resnet18(pretrained=True)

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
from torch.utils import model_zoo
from torch.optim import lr_scheduler

# #hyper parameter
batch_size = 32
num_epochs = 50
lr = 0.0001
num_classes = 25
image_size = 128 #### 64

path = "train"
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomRotation((30,30)),
    transforms.RandomVerticalFlip(0.1),
    transforms.RandomGrayscale(0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

dataset = datasets.ImageFolder(path, transform=transform)

print("dataset.classes",dataset.classes)
print("dataset.class_to_idx",dataset.class_to_idx)
idx_to_class = dict((v, k) for k, v in dataset.class_to_idx.items())
print("idx_to_class",idx_to_class)
print('len(dataset)', len(dataset))

"""将训练集划分为训练集和验证集"""
train_db, val_db = torch.utils.data.random_split(dataset, [2050, 325])#####
print('train:', len(train_db), 'validation:', len(val_db))

# 训练集
train_loader = torch.utils.data.DataLoader(
    train_db,
    batch_size=batch_size,
    shuffle=True, 
    drop_last=False)
test_loader = torch.utils.data.DataLoader(
    val_db,
    batch_size=batch_size,
    shuffle=True)

classes = set(['Seashell', 'Lighter', 'Old Mirror', 'Broom', 'Ceramic Bowl', 'Toothbrush', 'Disposable Chopsticks', 'Dirty Cloth',
'Newspaper', 'Glassware', 'Basketball', 'Plastic Bottle', 'Cardboard', 'Glass Bottle', 'Metalware', 'Hats', 'Cans', 'Paper',
'Vegetable Leaf', 'Orange Peel', 'Eggshell', 'Banana Peel',
'Battery', 'Tablet capsules', 'Paint bucket'])


# declare and define an objet of MyCNN
mycnn = MyCNN(image_size, num_classes)
print(mycnn)
# mycnn = getRsn()
# mycnn = getMbnet()
# os.environ['PATH']

# x = torch.randn(16, 3, 224, 224).requires_grad_(True)  # 定义一个网络的输入值
# y = mycnn(x)    # 获取网络的预测值

# MyConvNetVis = make_dot(y, params=dict(list(mycnn.named_parameters()) + [('x', x)]))
# MyConvNetVis.format = "png"
# # 指定文件生成的文件夹
# MyConvNetVis.directory = "result"
# # 生成文件
# MyConvNetVis.view()

# device = torch.device('cuda:0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizer = torch.optim.Adam(mycnn.parameters(), lr=lr)

# start training on cifar10 dataset
fit(mycnn, num_epochs, optimizer, device)
```

##### MobileNetV2

```python
# Temp 3
import math
import numpy as np
import os
import cv2
import random
import shutil
import time
from matplotlib import pyplot as plt
from easydict import EasyDict
from PIL import Image

import mindspore as ms
from mindspore import context
from mindspore import nn
from mindspore import Tensor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, save_checkpoint, export
from mindspore.train.callback import Callback, LossMonitor, ModelCheckpoint, CheckpointConfig

from src_mindspore.dataset import create_dataset # 数据处理脚本
from src_mindspore.mobilenetv2 import MobileNetV2Backbone, mobilenet_v2 # 模型定义脚本

os.environ['GLOG_v'] = '2' # Log Level = Error
has_gpu = (os.system('command -v nvidia-smi') == 0)
print('Excuting with', 'GPU' if has_gpu else 'CPU', '.')
context.set_context(mode=context.GRAPH_MODE, device_target='GPU' if has_gpu else 'CPU')

# 垃圾分类数据集标签，以及用于标签映射的字典。
index = {'00_00': 0, '00_01': 1, '00_02': 2, '00_03': 3, '00_04': 4, '00_05': 5, '00_06': 6, '00_07': 7,
         '00_08': 8, '00_09': 9, '01_00': 10, '01_01': 11, '01_02': 12, '01_03': 13, '01_04': 14, 
         '01_05': 15, '01_06': 16, '01_07': 17, '02_00': 18, '02_01': 19, '02_02': 20, '02_03': 21,
         '03_00': 22, '03_01': 23, '03_02': 24, '03_03': 25}
inverted = {0: 'Plastic Bottle', 1: 'Hats', 2: 'Newspaper', 3: 'Cans', 4: 'Glassware', 5: 'Glass Bottle', 6: 'Cardboard', 7: 'Basketball',
            8: 'Paper', 9: 'Metalware', 10: 'Disposable Chopsticks', 11: 'Lighter', 12: 'Broom', 13: 'Old Mirror', 14: 'Toothbrush',
            15: 'Dirty Cloth', 16: 'Seashell', 17: 'Ceramic Bowl', 18: 'Paint bucket', 19: 'Battery', 20: 'Fluorescent lamp', 21: 'Tablet capsules',
            22: 'Orange Peel', 23: 'Vegetable Leaf', 24: 'Eggshell', 25: 'Banana Peel'}

# 训练超参
config = EasyDict({
    "num_classes": 26, # 分类数，即输出层的维度
    "reduction": 'max', # mean, max, Head部分池化采用的方式
    "image_height": 224,
    "image_width": 224,
    "batch_size": 26, # 鉴于CPU容器性能，太大可能会导致训练卡住
    "eval_batch_size": 15,
    "epochs": 35, # 请尝试修改以提升精度
    "lr_max": 0.003, # 请尝试修改以提升精度
    "decay_type": 'square', # 请尝试修改以提升精度
    "momentum": 0.9, # 请尝试修改以提升精度
    "weight_decay": 2.0, # 请尝试修改以提升精度
    "dataset_path": "./datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100",
    "features_path": "./results/garbage_26x100_features", # 临时目录，保存冻结层Feature Map，可随时删除
    "class_index": index,
    "save_ckpt_epochs": 1,
    "save_ckpt_path": './results/ckpt_mobilenetv2',
    "pretrained_ckpt": './src_mindspore/mobilenetv2-200_1067_cpu_gpu.ckpt',
    "export_path": './results/mobilenetv2.mindir'
    
})

ds = create_dataset(config=config, training=False)
data = ds.create_dict_iterator(output_numpy=True).get_next()
images = data['image']
labels = data['label']

for i in range(1, 5):
    plt.subplot(2, 2, i)
    plt.imshow(np.transpose(images[i], (1,2,0)))
    plt.title('label: %s' % inverted[labels[i]])
    plt.xticks([])
plt.show()

def build_lr(total_steps, lr_init=0.0, lr_end=0.0, lr_max=0.1, warmup_steps=0, decay_type='cosine'):
    """
    Applies cosine decay to generate learning rate array.

    Args:
       total_steps(int): all steps in training.
       lr_init(float): init learning rate.
       lr_end(float): end learning rate
       lr_max(float): max learning rate.
       warmup_steps(int): all steps in warmup epochs.

    Returns:
       list, learning rate array.
    """
    lr_init, lr_end, lr_max = float(lr_init), float(lr_end), float(lr_max)
    decay_steps = total_steps - warmup_steps
    lr_all_steps = []
    inc_per_step = (lr_max - lr_init) / warmup_steps if warmup_steps else 0
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + inc_per_step * (i + 1)
        else:
            if decay_type == 'cosine':
                cosine_decay = 0.5 * (1 + math.cos(math.pi * (i - warmup_steps) / decay_steps))
                lr = (lr_max - lr_end) * cosine_decay + lr_end
            elif decay_type == 'square':
                frac = 1.0 - float(i - warmup_steps) / (total_steps - warmup_steps)
                lr = (lr_max - lr_end) * (frac * frac) + lr_end
            else:
                lr = lr_max
        lr_all_steps.append(lr)

    return lr_all_steps

steps = 5*93
plt.plot(range(steps), build_lr(steps, lr_max=0.1, decay_type='constant'))
plt.plot(range(steps), build_lr(steps, lr_max=0.1, decay_type='square'))
plt.plot(range(steps), build_lr(steps, lr_max=0.1, decay_type='cosine'))
plt.show()

def extract_features(net, dataset_path, config):
    if not os.path.exists(config.features_path):
        os.makedirs(config.features_path)
    dataset = create_dataset(config=config)
    step_size = dataset.get_dataset_size()
    if step_size == 0:
        raise ValueError("The step_size of dataset is zero. Check if the images count of train dataset is more \
            than batch_size in config.py")

    data_iter = dataset.create_dict_iterator()
    for i, data in enumerate(data_iter):
        features_path = os.path.join(config.features_path, f"feature_{i}.npy")
        label_path = os.path.join(config.features_path, f"label_{i}.npy")
        if not os.path.exists(features_path) or not os.path.exists(label_path):
            image = data["image"]
            label = data["label"]
            features = net(image)
            np.save(features_path, features.asnumpy())
            np.save(label_path, label.asnumpy())
        print(f"Complete the batch {i+1}/{step_size}")
    return

backbone = MobileNetV2Backbone()
load_checkpoint(config.pretrained_ckpt, net=backbone)
extract_features(backbone, config.dataset_path, config)

class GlobalPooling(nn.Cell):
    """
    Global avg pooling definition.

    Args:
        reduction: mean or max, which means AvgPooling or MaxpPooling.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> GlobalAvgPooling()
    """

    def __init__(self, reduction='mean'):
        super(GlobalPooling, self).__init__()
        if reduction == 'max':
            self.mean = ms.ops.ReduceMax(keep_dims=False)
        else:
            self.mean = ms.ops.ReduceMean(keep_dims=False)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x


class MobileNetV2Head(nn.Cell):
    """
    MobileNetV2Head architecture.

    Args:
        input_channel (int): Number of channels of input.
        hw (int): Height and width of input, 7 for MobileNetV2Backbone with image(224, 224).
        num_classes (int): Number of classes. Default is 1000.
        reduction: mean or max, which means AvgPooling or MaxpPooling.
        activation: Activation function for output logits.
    Returns:
        Tensor, output tensor.

    Examples:
        >>> MobileNetV2Head(num_classes=1000)
    """

    def __init__(self, input_channel=1280, hw=7, num_classes=1000, reduction='mean', activation="None"):
        super(MobileNetV2Head, self).__init__()
        if reduction:
            self.flatten = GlobalPooling(reduction)
        else:
            self.flatten = nn.Flatten()
            input_channel = input_channel * hw * hw
        self.dense = nn.Dense(input_channel, num_classes, weight_init='ones', has_bias=False)
        if activation == "Sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "Softmax":
            self.activation = nn.Softmax()
        else:
            self.need_activation = False

    def construct(self, x):
        x = self.flatten(x)
        x = self.dense(x)
        if self.need_activation:
            x = self.activation(x)
        return x

def train_head():
    train_dataset = create_dataset(config=config)
    eval_dataset = create_dataset(config=config)
    step_size = train_dataset.get_dataset_size()
    
    backbone = MobileNetV2Backbone()
    # Freeze parameters of backbone. You can comment these two lines.
    for param in backbone.get_parameters():
       param.requires_grad = False
    load_checkpoint(config.pretrained_ckpt, net=backbone)

    head = MobileNetV2Head(input_channel=backbone.out_channels, num_classes=config.num_classes, reduction=config.reduction)
    network = mobilenet_v2(backbone, head)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    lrs = build_lr(config.epochs * step_size, lr_max=config.lr_max, warmup_steps=0, decay_type=config.decay_type)
    opt = nn.Momentum(head.trainable_params(), lrs, config.momentum, config.weight_decay)
    net = nn.WithLossCell(head, loss)
    train_step = nn.TrainOneStepCell(net, opt)
    train_step.set_train()
    
    # train
    history = list()
    features_path = config.features_path
    idx_list = list(range(step_size))
    for epoch in range(config.epochs):
        random.shuffle(idx_list)
        epoch_start = time.time()
        losses = []
        for j in idx_list:
            feature = Tensor(np.load(os.path.join(features_path, f"feature_{j}.npy")))
            label = Tensor(np.load(os.path.join(features_path, f"label_{j}.npy")))
            losses.append(train_step(feature, label).asnumpy())
        epoch_seconds = (time.time() - epoch_start)
        epoch_loss = np.mean(np.array(losses))
        
        history.append(epoch_loss)
        print("epoch: {}, time cost: {}, avg loss: {}".format(epoch + 1, epoch_seconds, epoch_loss))
        if (epoch + 1) % config.save_ckpt_epochs == 0:
            save_checkpoint(network, os.path.join(config.save_ckpt_path, f"mobilenetv2-{epoch+1}.ckpt"))
    
    # evaluate
    print('validating the model...')
    eval_model = Model(network, loss, metrics={'acc', 'loss'})
    acc = eval_model.eval(eval_dataset, dataset_sink_mode=False)
    print(acc)
    
    return history

if os.path.exists(config.save_ckpt_path):
    shutil.rmtree(config.save_ckpt_path)
os.makedirs(config.save_ckpt_path)

history = train_head()

plt.plot(history, label='train_loss')
plt.legend()
plt.show()

CKPT = f'mobilenetv2-{config.epochs}.ckpt'
print("Chosen checkpoint is", CKPT)

def image_process(image):
    """Precess one image per time.
    
    Args:
        image: shape (H, W, C)
    """
    mean=[0.485*255, 0.456*255, 0.406*255]
    std=[0.229*255, 0.224*255, 0.225*255]
    image = (np.array(image) - mean) / std
    image = image.transpose((2,0,1))
    img_tensor = Tensor(np.array([image], np.float32))
    return img_tensor

def infer_one(network, image_path):
    image = Image.open(image_path).resize((config.image_height, config.image_width))
    logits = network(image_process(image))
    pred = np.argmax(logits.asnumpy(), axis=1)[0]
    print(image_path, inverted[pred])
    return pred

def infer(images):
    backbone = MobileNetV2Backbone()
    head = MobileNetV2Head(input_channel=backbone.out_channels, num_classes=config.num_classes, reduction=config.reduction)
    network = mobilenet_v2(backbone, head)
    print('加载模型路径:',os.path.join(config.save_ckpt_path, CKPT))
    load_checkpoint(os.path.join(config.save_ckpt_path, CKPT), net=network)
    for img in images:
        infer_one(network, img)

test_images = list()
folder = os.path.join(config.dataset_path, 'val/00_08') # Hats
for img in os.listdir(folder):
    test_images.append(os.path.join(folder, img))

infer(test_images)

backbone = MobileNetV2Backbone()
# 导出带有Softmax层的模型
head = MobileNetV2Head(input_channel=backbone.out_channels, num_classes=config.num_classes,
                       reduction=config.reduction, activation='Softmax')
network = mobilenet_v2(backbone, head)
load_checkpoint(os.path.join(config.save_ckpt_path, CKPT), net=network)

input = np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]).astype(np.float32)
export(network, Tensor(input), file_name=config.export_path, file_format='MINDIR')

## 生成 main.py 时请勾选此 cell
# 本示范以 NoteBook 训练模型通过平台测试为例：

# 1. 导入相关包
import os
import cv2
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from easydict import EasyDict
from mindspore import context
from mindspore.train.serialization import load_checkpoint
from src_mindspore.mobilenetv2 import MobileNetV2Backbone, mobilenet_v2  # 模型定义脚本

os.environ['GLOG_v'] = '2'  # Log Level = Error
has_gpu = (os.system('command -v nvidia-smi') == 0)
print('Excuting with', 'GPU' if has_gpu else 'CPU', '.')
context.set_context(mode=context.GRAPH_MODE, device_target='GPU' if has_gpu else 'CPU')

# 2.系统测试部分标签与该处一致，请不要改动
# 垃圾分类数据集标签，以及用于标签映射的字典。
index = {'00_00': 0, '00_01': 1, '00_02': 2, '00_03': 3, '00_04': 4, '00_05': 5, '00_06': 6, '00_07': 7,
         '00_08': 8, '00_09': 9, '01_00': 10, '01_01': 11, '01_02': 12, '01_03': 13, '01_04': 14,
         '01_05': 15, '01_06': 16, '01_07': 17, '02_00': 18, '02_01': 19, '02_02': 20, '02_03': 21,
         '03_00': 22, '03_01': 23, '03_02': 24, '03_03': 25}
inverted = {0: 'Plastic Bottle', 1: 'Hats', 2: 'Newspaper', 3: 'Cans', 4: 'Glassware', 5: 'Glass Bottle', 6: 'Cardboard', 7: 'Basketball',
            8: 'Paper', 9: 'Metalware', 10: 'Disposable Chopsticks', 11: 'Lighter', 12: 'Broom', 13: 'Old Mirror', 14: 'Toothbrush',
            15: 'Dirty Cloth', 16: 'Seashell', 17: 'Ceramic Bowl', 18: 'Paint bucket', 19: 'Battery', 20: 'Fluorescent lamp', 21: 'Tablet capsules',
            22: 'Orange Peel', 23: 'Vegetable Leaf', 24: 'Eggshell', 25: 'Banana Peel'}

## 生成 main.py 时请勾选此 cell

# 3. NoteBook 模型调整参数部分，你可以根据自己模型需求修改、增加、删除、完善部分超参数
# 训练超参
config = EasyDict({
    "num_classes": 26,
    "reduction": 'mean',
    "image_height": 224,
    "image_width": 224,
    "eval_batch_size": 10
})

# 4. 自定义模型Head部分
class GlobalPooling(nn.Cell):
    def __init__(self, reduction='mean'):
        super(GlobalPooling, self).__init__()
        if reduction == 'max':
            self.mean = ms.ops.ReduceMax(keep_dims=False)
        else:
            self.mean = ms.ops.ReduceMean(keep_dims=False)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x


class MobileNetV2Head(nn.Cell):
    def __init__(self, input_channel=1280, hw=7, num_classes=1000, reduction='mean', activation="None"):
        super(MobileNetV2Head, self).__init__()
        if reduction:
            self.flatten = GlobalPooling(reduction)
        else:
            self.flatten = nn.Flatten()
            input_channel = input_channel * hw * hw
        self.dense = nn.Dense(input_channel, num_classes, weight_init='ones', has_bias=False)
        if activation == "Sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "Softmax":
            self.activation = nn.Softmax()
        else:
            self.need_activation = False

    def construct(self, x):
        x = self.flatten(x)
        x = self.dense(x)
        if self.need_activation:
            x = self.activation(x)
        return x


# -------------------------- 5.请加载您最满意的模型 ---------------------------
# 首先加载网络模型
backbone = MobileNetV2Backbone()
head = MobileNetV2Head(input_channel=backbone.out_channels, num_classes=config.num_classes, reduction=config.reduction)
network = mobilenet_v2(backbone, head)

# 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
# 如果你的模型是在 results 文件夹下的模型，则 model_path = './results/ckpt_mobilenetv2/mobilenetv2-4.ckpt'

model_path = './results/ckpt_mobilenetv2/mobilenetv2-4.ckpt'
load_checkpoint(model_path, net=network)

# ---------------------------------------------------------------------------

def image_process(image):
    """Precess one image per time.
    
    Args:
        image: shape (H, W, C)
    """
    mean=[0.485*255, 0.456*255, 0.406*255]
    std=[0.229*255, 0.224*255, 0.225*255]
    image = (np.array(image) - mean) / std
    image = image.transpose((2,0,1))
    img_tensor = Tensor(np.array([image], np.float32))
    return img_tensor

def predict(image):
    """
    加载模型和模型预测
    主要步骤:
        1.图片处理,此处尽量与训练模型数据处理一致
        2.用加载的模型预测图片的类别
    :param image: OpenCV 读取的图片对象，数据类型是 np.array，shape (H, W, C)
    :return: string, 模型识别图片的类别, 
            包含 'Plastic Bottle','Hats','Newspaper','Cans'等共 26 个类别
    """
    # -------------------------- 实现图像处理部分的代码 ---------------------------
    # 该处是与 NoteBook 训练数据预处理一致；
    # 如使用其它方式进行数据处理，请修改完善该处，否则影响成绩
    image = cv2.resize(image,(config.image_height, config.image_width))
    image = image_process(image)
    
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    logits = network(image)
    pred = np.argmax(logits.asnumpy(), axis=1)[0]
    
    return inverted[pred]

# 输入图片路径和名称
image_path = './datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100/val/00_00/00037.jpg'

# 使用 opencv 读取图片
image = cv2.imread(image_path)

# 打印返回结果
print(predict(image))
```

