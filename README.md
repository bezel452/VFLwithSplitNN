# VFL with SplitNN
CS in Xi‘an Jiao Tong University

## Description

It is the repository of code of the simulation of multi-participant's VFL with Split Neural Network.

## Manual

包含几个部分

* `data`包含数据集加载以及预处理（切分特征）的部分
* `datasets`包含数据集在服务器中的地址
* `networks`包含数据集训练使用的网络
* `parties`包含参与联邦学习的客户端与服务端的类设计
* `recipe`包含创建模型以及训练使用的脚本

主程序`VSL.py`包含几个选项

* `-n, --num_client`设置客户端（参与者）的数量
* `-d, --dataset`设置训练使用的数据集（目前只支持CIFAR-10，CIFAR-100，CINIC-10，BHI，Image-Nette）
* `-e, --epochs`设置训练的迭代次数
* `-b, --batch_size`设置训练集的batch size
* `-l, --learning_rate`设置训练的学习率
* `-m, --momentum`设置训练的动量

默认值`python VSL.py -n 2 -d cifar10 -e 20 -b 128 -l 0.01 -m 0.9`

## Others

はいはい、これはセイアンコーツー大学　コンピューターサイエンス大学院　の　大学院生　bezelです。

ここは「Vertical Federated Learning のシミュレーション」と、コードストレージするリポジトリです。

それにしても、どうな言語は難しいですねー、コンピューターのも人類のも。

