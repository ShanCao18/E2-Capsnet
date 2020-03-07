import numpy as np
import os
from Dataset_ import VideoDataset
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data as Data
from tensorboardX import SummaryWriter

writer = SummaryWriter('enet16_norm_2map_cap')

USE_CUDA = True

batch_size = 16
n_epochs = 300
NUM_ROUTING_ITERATIONS = 3
NUM_CLASSES = 7

class average_meter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_to():
    test_acc = './test_acc.csv'
    if os.path.exists(test_acc):
        os.remove(test_acc)
    fd_test_acc = open(test_acc, 'w')
    fd_test_acc.write('test_acc\n')
    return fd_test_acc


class RAF:
    def __init__(self, batch_size):

        transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        transforms_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        train_dataset = VideoDataset(root='./RAF/', list='train_list.txt', train_test='train', transform=transforms_train)
        self.test_dataset = VideoDataset(root='./RAF/', list='test_list.txt', train_test='test', transform=transforms_test)

        self.train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=2, pin_memory=True, drop_last=True)
        self.test_loader = Data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=2, pin_memory=True, drop_last=True)


class ENet16(nn.Module):
    def __init__(self):
        super(ENet16, self).__init__()
        self.layer_feature = nn.Sequential(

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer_feature2 = nn.Sequential(

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )

        self.layer1 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 1-2 conv layer
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 1 Pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2))   # 2x64x112x112

        self.layer2 = nn.Sequential(

            # 2-1 conv layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 2-2 conv layer
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 2 Pooling lyaer
            nn.MaxPool2d(kernel_size=2, stride=2))   # 2x128x56x56

        self.conv3_1 = nn.Sequential(

            # 3-1 conv layer
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),    # 2x256x56x56

        )

        self.layer3 = nn.Sequential(

            # 3-1 conv layer
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 3-2 conv layer
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 3-3 conv layer
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.layer3_pooling = nn.Sequential(

            nn.MaxPool2d(kernel_size=2, stride=2)

        )

        self.conv4_1 = nn.Sequential(

            # 4-1 conv layer
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.layer4 = nn.Sequential(

            # 4-1 conv layer
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # 4-2 conv layer
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # 4-3 conv layer
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.layer4_pooling = nn.Sequential(
            # 4 Pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(

            # 5-1 conv layer
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # 5-2 conv layer
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # 5-3 conv layer
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # 5 Pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # self.upsample = nn.Upsample(scale_factor=4, mode='nearest')

    def forward(self, x):
        feat = x[:, 3:4, :, :]
        feat = feat.type(torch.FloatTensor)
        feat = feat.cuda()

        feat1 = self.layer_feature(feat)   # 2x4x224x224
        map56x256 = feat1.expand(batch_size, 256, 56, 56)

        feat2 = self.layer_feature2(feat)
        map28x512 = feat2.expand(batch_size, 512, 28, 28)   # 16x512x28x28

        # print(type(x[:, 0:3, :, :]))
        data = x[:, 0:3, :, :]
        data = data.type(torch.FloatTensor)
        data = data.cuda()
        # print(data)

        out = self.layer1(data)  # 16x64x112x112
        layer2 = self.layer2(out)  # 16x128x56x56
        out = self.layer3(layer2)   # 16x256x56x56

        conv3_1 = self.conv3_1(layer2)       # 16x256x56x56
        conv3_map = map56x256 * conv3_1      # 16x256x56x56
        out = out + conv3_map                # 16x256x56x56
        out = self.layer3_pooling(out)       # 16x256x28x28

        conv4_1 = self.conv4_1(out)         # 16x512x28x28
        conv4_map = map28x512 * conv4_1     # 16x512x28x28
        out = self.layer4(out)              # 16x512x28x28
        out = out + conv4_map               # 16x512x28x28
        out = self.layer4_pooling(out)      # 16x512x14x14

        out = self.layer5(out)              # 16x512x7x7
        # out = self.upsample(out)
        return out


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.enet16 = ENet16()
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=512, out_channels=32,
                                             kernel_size=2, stride=1)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)

        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 150528),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = self.enet16(x)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices)

        reconstructions = self.decoder((x * masked[:, :, None]).view(x.size(0), -1))

        return masked, classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        # self.reconstruction_loss = nn.MSELoss(size_average=False)
        self.reconstruction_loss = nn.MSELoss()


    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()
        # assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


capsule_net = CapsuleNet()
capsule_loss = CapsuleLoss()

if USE_CUDA:
    capsule_net = capsule_net.cuda()
optimizer = Adam(capsule_net.parameters(), lr=1e-4)  # lr=1e-4

raf = RAF(batch_size)
fd_test_acc = save_to()


def train(epoch):
    losses = average_meter()
    accuracy = average_meter()
    capsule_net.train()

    for batch_id, (data, label) in enumerate(raf.train_loader):
        # print(label)  # torch.LongTensor

        target = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=label)

        data, target = Variable(data).cuda(), Variable(target).cuda()

        masked, output, reconstructions = capsule_net(data)

        data = data[:, 0:3, :, :]
        data = data.type(torch.FloatTensor)
        data = data.cuda()
        loss = capsule_loss(data, target, output, reconstructions)
        losses.update(loss.data[0], data.size(0))

        pred = output.data.max(1)[1]
        # print(pred) #  torch.cuda.LongTensor
        label = label.type(torch.LongTensor)
        label = Variable(label).cuda()
        # prec = pred.eq(target.data).cpu().sum()
        prec = pred.eq(label.data).cpu().sum()

        accuracy.update(float(prec) / data.size(0), data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_id+1) % 100 == 0:
            print('Epoch [%d/%d], Iter[%d/%d], train loss. %.4f train accuracy: %.4f' %
                  (epoch, n_epochs, batch_id + 1, len(raf.train_loader),
                  losses.val, accuracy.val))

        writer.add_scalar('Train/Loss', losses.val, epoch)
        writer.add_scalar('Train/Acc', accuracy.val, epoch)


def test(epoch):
    losses = average_meter()
    accuracy = average_meter()

    capsule_net.eval()

    mat = np.zeros((1, 7))
    for data, label in raf.test_loader:
        target = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=label)

        data, target = Variable(data).cuda(), Variable(target).cuda()

        masked, output, reconstructions = capsule_net(data)

        data = data[:, 0:3, :, :]
        data = data.type(torch.FloatTensor)
        data = data.cuda()
        loss = capsule_loss(data, target, output, reconstructions)
        losses.update(loss.data[0], data.size(0))

        pred = output.data.max(1)[1]
        label = label.type(torch.LongTensor)
        label = Variable(label).cuda()
        prec = pred.eq(label.data).cpu().sum()

        accuracy.update(float(prec) / data.size(0), data.size(0))

        mat = np.vstack((mat, output.cpu().data.view(batch_size, -1).numpy()))

    df = pd.DataFrame(mat[1:])
    df.to_excel('enet_cap.xlsx')

    print('\nTest: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        losses.avg, int(accuracy.sum), len(raf.test_dataset), 100. * accuracy.avg))

    writer.add_scalar('Test/Loss', losses.avg, epoch)
    writer.add_scalar('Test/Acc', accuracy.avg, epoch)

    return accuracy.avg


def main():
    best_model = capsule_net
    best_accuray = 0.0

    for epoch in range(1, n_epochs):

        train(epoch)
        val_accuracy = test(epoch)

        if best_accuray < val_accuracy:
            best_model = capsule_net
            best_accuray = val_accuracy

        fd_test_acc.close()

    writer.close()

    print("The best model has an accuracy of " + str(best_accuray))
    torch.save(best_model.state_dict(), 'E2Capsnet.model')


if __name__ == '__main__':
    main()


