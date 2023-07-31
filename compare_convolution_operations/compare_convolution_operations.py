# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:47:37 2023

@author: Manish Sharma
"""

### Compare different convolution operations ###
# Standard
# Depthwise separable: Depthwise + Pointwise
# Spatially separable: Horizonal + Vertical
# Spatially depthwise separable: Horizonal(Depthwise) + Vertical(Depthwise) + Pointwise
# Dilated

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from thop import profile
import pandas as pd
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# Fix seeds for all sources of randomness
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check if CUDA is available and set PyTorch to use GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def top_k_accuracy(output, target, k=5):
    batch_size = target.size(0)

    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / batch_size)

# Convolutional Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, conv_type='standard'):
        super(ConvBlock, self).__init__()
        if conv_type == 'standard':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        elif conv_type == 'depthwise_separable':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, out_channels, 1, bias=bias)
            )
        elif conv_type == 'spatially_separable':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, (1, kernel_size), (1, stride), (0, 1), bias=False),
                nn.Conv2d(in_channels, out_channels, (kernel_size, 1), (stride, 1), (1, 0), bias=bias)
            )
        elif conv_type == 'spatially-depthwise_separable':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, (1, kernel_size), (1, stride), (0, 1), groups=in_channels, bias=False),
                nn.Conv2d(in_channels, in_channels, (kernel_size, 1), (stride, 1), (1, 0), groups=in_channels, bias=False),
                nn.Conv2d(in_channels, out_channels, 1, bias=bias)
            )
        elif conv_type == 'dilated':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 1, dilation=2, bias=bias)
        else:
            raise ValueError('Invalid conv_type')
        
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

# Network Architecture
class Net(nn.Module):
    def __init__(self, conv_type):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 32, 3, 2, 1, conv_type=conv_type),
            ConvBlock(32, 64, 3, 2, 1, conv_type=conv_type),
            ConvBlock(64, 128, 3, 2, 1, conv_type=conv_type),
            ConvBlock(128, 256, 3, 2, 1, conv_type=conv_type),
            ConvBlock(256, 10, 2, 1, 0, bias=True, conv_type=conv_type),  # Convert to 10 classes
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)

# Load the CIFAR10 dataset
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    AutoAugment(AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(root='../data/cifar10', train=True, download=True, transform=train_transform)
testset = datasets.CIFAR10(root='../data/cifar10', train=False, download=True, transform=test_transform)
trainloader = DataLoader(trainset, batch_size=2500, shuffle=True)
testloader = DataLoader(testset, batch_size=2500, shuffle=False)

# Create models with different convolution types
conv_types = ['standard', 'depthwise_separable', 'spatially_separable', 'spatially-depthwise_separable'] #, 'dilated']

# Dataframes
dt = pd.DataFrame(columns=['Conv Type', 'Epoch', 'Lr', 'Params', 'FLOPs',
#                            'Train Loss', 'Test Loss', 
                           'Train Accuracy', 'Test Accuracy',
#                            'Train Top-5 Accuracy', 'Test Top-5 Accuracy',
#                            'Train Time', 'Test Time',
                          ])
df_dict = {}


for conv_type in conv_types:
    df = pd.DataFrame(columns=dt.columns)
    model = Net(conv_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)
    # print(model)
    # Count parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Measure FLOPs
    input = torch.randn(1, 3, 32, 32).to(device)
    flops, _ = profile(model, inputs=(input, ), verbose=False)
    
    # Training loop
    for epoch in range(10):
        train_correct, train_correct_k, train_total, train_loss = 0, 0, 0, 0.0
        model.train()
        start_train_time = time.time()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            train_correct_k += top_k_accuracy(outputs, labels)
            train_loss += loss.item()

        scheduler.step(loss)
        end_train_time = time.time()

        # Get current learning rate
        lr = optimizer.param_groups[0]['lr']

        # Testing loop
        start_test_time = time.time()
        test_correct, test_correct_k, test_total, test_loss = 0, 0, 0, 0.0
        model.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                test_correct_k += top_k_accuracy(outputs, labels)
                test_loss += criterion(outputs, labels).item()

        end_test_time = time.time()

        # Create new data and append to dataframe
        new_data = pd.DataFrame({
            'Conv Type': conv_type,
            'Epoch': epoch+1,
            'Lr': lr,
            'Params': params,
            'FLOPs': int(flops),
#             'Train Loss': train_loss / len(trainloader),
#             'Test Loss': test_loss / len(testloader),
            'Train Accuracy': round(100. * train_correct / train_total, 2),
            'Test Accuracy': round(100. * test_correct / test_total, 2),
#             'Train Top-5 Accuracy': train_correct_k.cpu().numpy() / len(trainloader),
#             'Test Top-5 Accuracy': test_correct_k.cpu().numpy() / len(testloader),
#             'Training Time': end_train_time - start_train_time,
#             'Testing Time': end_test_time - start_test_time
        }, index=[0])
        df = pd.concat([df, new_data], ignore_index=True)
    
    dt = pd.concat([dt, df.iloc[[-1]]], ignore_index=True)
    df_dict[conv_type] = df
    
    # Clean up
    del model
    del optimizer
    del scheduler
    torch.cuda.empty_cache()

# Display final data
print(dt)
