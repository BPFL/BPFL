import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from model import *
from util import *

def test_model(model_name,model_weight,test_data,device="cpu"):
    if model_name=="cnn":
        model=CNN()
    elif model_name=="lenet":
        model=LeNet()
    elif model_name=="resnet":
        model=ResNet20()
    else:
        print("parameter error")
        return
    model.load_state_dict(model_weight)
    test_dataloader = DataLoader(test_data, batch_size=64)
    test_data_size = len(test_data)
    model.to(device)
    model.eval()
    total_test_loss = 0
    total_accracy = 0
    total_test_step = 0

    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    with torch.no_grad():
        for data in test_dataloader:
            img, target = data
            img = img.to(device)
            target = target.to(device)
            output = model(img)
            loss = loss_fn(output, target)
            total_test_loss += loss
            accracy = (output.argmax(1) == target).sum()
            total_accracy += accracy
        total_test_step += 1
    torch.cuda.empty_cache()
    return (total_test_loss.item(),total_accracy.item() / test_data_size)

def train(model_name,model_weight,train_data,device="cpu",lr=0.1,epoch=5,dp=0):
    if model_name=="cnn":
        model=CNN()
    elif model_name=="lenet":
        model=LeNet()
    elif model_name=="resnet":
        model=ResNet20()
    else:
        print("parameter error")
        return
    model.load_state_dict(model_weight)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    optim = torch.optim.SGD(model.parameters(), lr)
    train_dataloader = DataLoader(train_data, batch_size=len(train_data))
    for i in range(epoch):
        model.train()
        for data in train_dataloader:
            img, target = data
            img = img.to(device)
            target = target.to(device)
            output = model(img)
            loss = loss_fn(output, target)
            optim.zero_grad()
            loss.backward()
            if dp!=0 and i==epoch-1:
                sensitivity = cal_sensitivity(lr, 10, len(train_data))
                for k, v in model.named_parameters():
                    v.grad /= max(1, v.grad.norm(2) / 10)
                    noise = Laplace(epsilon=10,sensitivity=sensitivity, size=v.size())
                    noise = torch.from_numpy(noise).to(device)
                    v.grad+=noise
            optim.step()
    torch.cuda.empty_cache()
    return model.state_dict()