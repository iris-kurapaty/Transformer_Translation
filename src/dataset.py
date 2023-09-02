import numpy
import torch
import torchvision
from torch.utils.data import DataLoader

class cifar10Dataset(torchvision.datasets.CIFAR10):
  """"
  Custom Dataset class
  """

  def __init__(self, root = "./data", train = True, transform = None, download = True):
    super().__init__(root=root, train=train, transform=transform, download=download)
    self.transform = transform

  def __getitem__(self, index):
    
    img, target = self.data[index], self.targets[index]

    if self.transform is not None:
      img = self.transform(image=img)["image"]
    return (img, target)

def get_loader(train_data, test_data, batch_size, use_cuda):

  dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if use_cuda else dict(shuffle=True, batch_size=64)

  # train dataloader
  train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)

  # test dataloader
  test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)

  return train_loader, test_loader