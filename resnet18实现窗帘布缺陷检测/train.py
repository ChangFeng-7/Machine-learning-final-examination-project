import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import  DataLoader,Dataset
import os
from PIL import Image
from torchvision import datasets
from utils.DealDataset import DealDataset
from resnet18 import ResNet18



# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size_train = 20  # 批处理尺寸(batch_size)
batch_size_test = 20  # 批处理尺寸(batch_size)
LR = 0.001   # 学习率


# 数据捕捉----------------------------------------------

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])


train_set = DealDataset(transforms = transform, datapath = "./data/train")
test_set = DealDataset(transforms = transform, datapath = "./data/test")


# 训练集
# trainset = torchvision.datasets.MNIST(
#     root = './data/',
#     train = True,
#     download = True,
#     transform = transform)
#

# 定义训练批处理数据
trainloader = torch.utils.data.DataLoader(
    train_set,
    batch_size = batch_size_train,
    shuffle = True)

# # 测试集
# testset = torchvision.datasets.MNIST(
#     root = './data/',
#     train = False,
#     download = True,
#     transform = transform)

# 定义测试批处理数据
testloader = torch.utils.data.DataLoader(
    test_set,
    batch_size = batch_size_test,
    shuffle = False)

# 定义损失函数和优化方式---------------------------------------
model = ResNet18(2).to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)


# 训练模型-------------------------------------------------------
epoch = 100

max_test_acc = 0
# model_save_path = "./save_model/"

model.train() # 模型激活训练
for epo in range(epoch):
    train_loss = 0.0
    train_acc = 0.0

    temp_count = 0
    for batch_idx, (data, target) in enumerate(trainloader):

        temp_count = temp_count+1
        print(temp_count)


        data, target = data.to(device), target.to(device)
        # optimizer.zero_grad()
        # data = data.permute([1,0,2,3])[0]
        # data = data.unsqueeze(0)
        # data = data.permute([1,0,2,3])

        output = model(data)
        _, preds = torch.max(output, 1)
        pre_ = target

        loss = criterion(output, target)
        # loss = criterion(preds, target)
        acc = torch.sum(preds == pre_)  # 判断对的个数

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # train_loss.append(loss.data)
        train_loss += (loss.cpu().detach().numpy())
        train_acc += (acc.cpu().detach().numpy())


    print('epoch: {}, Train loss:{:.6f}, Train acc:{:.6f}'.format(
        (epo+1), train_loss /len(trainloader)/ batch_size_train,
        train_acc/len(trainloader)/batch_size_train))




    eval_loss = 0.0
    eval_acc = 0.0

    model.eval()
    for batch_idx, (data, target) in enumerate(testloader):
        data, target = data.to(device), target.to(device)

        # data = data.permute([1,0,2,3])[0]
        # data = data.unsqueeze(0)
        # data = data.permute([1,0,2,3])

        output = model(data)
        _, preds = torch.max(output, 1)
        pre_ = target

        loss = criterion(output, target)  # 为什么不用preds??
        acc = torch.sum(preds == pre_)

        # train_loss.append(loss.data)
        eval_loss += loss.cpu().detach().numpy()
        eval_acc += acc.cpu().detach().numpy()




    print('Test loss:{:.6f}, Test acc:{:.6f}'.format(
        eval_loss /batch_size_test/len(testloader),
               eval_acc / batch_size_test/len(testloader)))

    if (eval_acc / batch_size_test/len(testloader)) >=  max_test_acc:
        max_test_acc = (eval_acc / batch_size_test/len(testloader))
        model_path = './save_model/%04d_acc_%.4f.pth' % (epo,max_test_acc)

        torch.save(model, model_path)

    # net = torch.load(args.pkl_path)  # 加载


