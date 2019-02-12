import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import visdom
import settings


class Flatten(nn.Module):
    def forward(self, batch):
        return batch.view(batch.size(0), -1)


class Unflatten(nn.Module):
    def forward(self, batch):
        return batch.view(batch.size(0), 8, 5, 5)


class Visualizer():
    def __init__(self):
        self.vis = visdom.Visdom()

    def update_batch(self, batch, name):
        self.vis.images(batch, win=name)

def get_dataloader():
    mnist, name = (datasets.MNIST, "MNIST") if not settings.fashion else \
            (datasets.FashionMNIST, "FashionMNIST")

    dataset = mnist("~/Data/{}".format(name), 
            train=True, 
            transform=transforms.ToTensor())

    return tdata.DataLoader(dataset, batch_size=settings.batch_size, shuffle=True, num_workers=4)
