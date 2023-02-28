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


from utils.DealDataset import DealDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

test_set = DealDataset(transforms = transform, datapath = "./data/test")
batch_size = 2

testloader = torch.utils.data.DataLoader(
    test_set,
    batch_size = 2,
    shuffle = False)

model_path = ".\save_model\\0025_acc_0.8667.pth"
# model = AlexNet().to(device)
model = torch.load(model_path)

model.eval()

criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上


eval_loss = 0.0
eval_acc = 0.0



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
    eval_loss / batch_size / len(testloader),
    eval_acc / batch_size / len(testloader)))

