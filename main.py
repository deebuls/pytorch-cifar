'''Train CIFAR10 with PyTorch.'''
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset


import torchvision
import torchvision.transforms as transforms

import os
import argparse

import numpy as np

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--neurips', '-n', action='store_true',
                    help='get neurips results  from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

hyper_params = {
    "num_classes": 10,
    "batch_size": 100,
    "num_epochs": 10,
    "learning_rate": args.lr,
    "num_workers": 2
}

# Create an experiment with your api key
experiment = Experiment(
    api_key="E8B7IntUqHCsJVXF9ZnPxK5UN",
    project_name="cifar10-neurips",
    workspace="deebuls",
)

experiment.log_parameters(hyper_params)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
#net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net =  get_model("resnet20_frn_swish", data_info={"num_classes": 10})

print (net)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume or args.neurips:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_cross_entropy.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print ('Loss: %.3f | Acc: %.3f%% (%d/%d) | Last LR: %.5f'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total, get_lr(optimizer)), flush=True)
    experiment.log_metric("train_accuracy", correct / total, step=epoch)
    experiment.log_metric("lr",get_lr(optimizer), step=epoch)
    experiment.log_metric("train loss", train_loss/(batch_idx+1), step=epoch)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print ('Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total), flush=True)
    experiment.log_metric("val_accuracy", correct / total, step=epoch)
    experiment.log_metric("lr",get_lr(optimizer), step=epoch)
    experiment.log_metric("test loss", test_loss/(batch_idx+1), step=epoch)
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..', acc)
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_cross_entropy.pth')
        best_acc = acc

def neurips_competition():
    x_test = np.loadtxt("/scratch/dnair2m/neurips_cifar_data/cifar10_test_x.csv")
    y_test = np.loadtxt("/scratch/dnair2m/neurips_cifar_data/cifar10_test_y.csv")
    
    x_test = x_test.reshape((len(x_test), 3, 32, 32))
    
    testset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    data_loader = DataLoader(testset, batch_size=100, shuffle=False)
    sum_accuracy = 0
    all_probs = []
    print ("starting loop")
    for x, y in data_loader:       
        if torch.cuda.is_available():
          x = x.cuda()
          y = y.cuda()
        # get logits 
        net.eval()
        with torch.no_grad():
          logits = net(x)
        net.train()
        # get log probs 
        log_probs = F.log_softmax(logits, dim=1)
        # get preds 
        batch_probs = torch.exp(log_probs)
        batch_preds = torch.argmax(logits, dim=1)
        batch_accuracy = (batch_preds == y).float().mean()
        sum_accuracy += batch_accuracy.item()
        all_probs.append(batch_probs)
    all_probs = torch.cat(all_probs, dim=0)
    print (" Test accuracy ", sum_accuracy/len(data_loader))
    return sum_accuracy / len(data_loader), all_probs


if args.neurips:
    neurips_competition()
else:
    
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
        scheduler.step()
