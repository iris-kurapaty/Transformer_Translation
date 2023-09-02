from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from tqdm import tqdm
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from dataset import cifar10Dataset, get_loader
BATCH_SIZE = 256

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CustomBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomBlock, self).__init__()

        self.inner_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.res_block = BasicBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.inner_layer(x)
        r = self.res_block(x)

        out = x + r

        return out
        
class LitNet(LightningModule):
  def __init__(self, max_lr):
    super().__init__()

    self.dropout = 0.1
    self.max_lr = max_lr

    self.criterion = nn.CrossEntropyLoss()
    self.accuracy = Accuracy(task='multiclass',num_classes=10)

    self.means = (0.4914, 0.4822, 0.4465)
    self.stds = (0.2470, 0.2435, 0.2616)

    self.train_transform =  A.Compose([
        A.Normalize(self.means, self.stds),
        A.PadIfNeeded(36,36),
        A.RandomCrop(height=32, width=32),
        A.HorizontalFlip(p=0.05),
        A.CoarseDropout(max_holes = 1, max_height=8, max_width=8, min_holes = 1, min_height=8, min_width=8, fill_value=self.means, mask_fill_value =None),
        ToTensorV2()])

    self.test_transform = A.Compose([
        A.Normalize(self.means, self.stds),
        ToTensorV2()])

    # Prep Layer input 32/1/1
    self.prep_layer = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU()
    ) # output_size =32

    self.layer_1 = CustomBlock(in_channels=64, out_channels=128)

    self.layer_2 = nn.Sequential(
        nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        nn.MaxPool2d(kernel_size=2),
        nn.BatchNorm2d(256),
        nn.ReLU(),
    )

    self.layer_3 = CustomBlock(in_channels=256, out_channels=512)

    self.max_pool = nn.Sequential(nn.MaxPool2d(kernel_size=4))

    self.fc = nn.Linear(512, 10)

  def forward(self, x):
      x = self.prep_layer(x)
      x = self.layer_1(x)
      x = self.layer_2(x)
      x = self.layer_3(x)
      x = self.max_pool(x)
      x = x.view(x.size(0), -1)
      x = self.fc(x)
      return x

  def training_step(self, batch, batch_idx):
      x, y = batch
      logits = self.forward(x)
      loss = self.criterion(logits, y)

      self.log("train_loss", loss, prog_bar=True)
      return loss

  def validation_step(self, batch, batch_idx):
      x, y = batch
      logits = self.forward(x)
      loss = self.criterion(logits, y)
      preds = torch.argmax(logits, dim=1)
      self.accuracy(preds, y)

      # Calling self.log will surface up scalars for you in TensorBoard
      self.log("val_loss", loss, prog_bar=True)
      self.log("val_acc", self.accuracy, prog_bar=True)
      return loss

  def test_step(self, batch, batch_idx):
      # Here we just reuse the validation_step for testing
      return self.validation_step(batch, batch_idx)

  def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=0.03, weight_decay=1e-4)
      num_epochs = self.trainer.max_epochs
      scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.max_lr,
                                                      steps_per_epoch=self.trainer.estimated_stepping_batches, epochs=num_epochs,
                                                      pct_start=5/num_epochs, div_factor=100,
                                                      three_phase=False,final_div_factor=100,anneal_strategy='linear')

      lr_scheduler = {"scheduler": scheduler, "interval": "step"}
      return [optimizer], [lr_scheduler]

  #####################################
  ## DATA RELATED
  #####################################

  def prepare_data(self):
      # download
      cifar10Dataset(root = "./data", train=True, download=True)
      cifar10Dataset(root = "./data", train=False, download=True)

  def setup(self, stage=None):


      # Assign train/val datasets for use in dataloaders
      if stage == "fit" or stage is None:
          self.full_data = cifar10Dataset(root = "./data", train=True, transform=self.train_transform)
          self.train_data, self.val_data = random_split(self.full_data, [45000, 5000])

      # Assign test dataset for use in dataloader(s)
      if stage == "test" or stage is None:
          self.test_data = cifar10Dataset(root = "./data", train=False, transform=self.test_transform)

  def train_dataloader(self):
      return DataLoader(self.train_data, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), shuffle=True, pin_memory=True)

  def val_dataloader(self):
      return DataLoader(self.val_data, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), shuffle=True, pin_memory=True)

  def test_dataloader(self):
      return DataLoader(self.test_data, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), shuffle=True, pin_memory=True)



#-----------------------------------------------------------------------------------------
# Model Classes from Assigment 10
#-----------------------------------------------------------------------------------------

# dropout_value = 0.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Prep Layer input 32/1/1
        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # output_size =32

        # Layer 1
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.ReLU()
            
        ) # output_size = 32

        self.res1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
            ) # output_size = 16

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ) # output_size = 16

        # Layer 3
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.ReLU()
            
        ) # output_size = 8

        self.res2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
            ) # output_size = 8

        # MAx Pooling with Kernel Size 4
        self.maxpool =  nn.MaxPool2d(4, 2)


        # fully connected layer
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.preplayer(x)
        x = self.convlayer1(x)
        r1 = self.res1(x)
        x = x + r1
        x = self.layer2(x)
        x = self.convlayer2(x)
        r2 = self.res2(x)
        x = x + r2
        x = self.maxpool(x)
        x = self.fc(torch.squeeze(x))
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Block(nn.Module):
  def __init__(self, input_size, out_size, drop_out):
    super(Block, self).__init__()
    self.drop_out = drop_out

    self.convblock1 = nn.Sequential(
        nn.Conv2d(in_channels=input_size, out_channels=out_size, kernel_size=(3,3), padding=1, bias=False),
        nn.BatchNorm2d(out_size),
        nn.Dropout(drop_out),
        nn.ReLU()
    )# output_size = 32; RF = 3

    self.convblock2 = nn.Sequential(
        nn.Conv2d(in_channels=out_size, out_channels = out_size, kernel_size=(3,3), padding=1, bias=False),
        nn.BatchNorm2d(out_size),
        nn.Dropout(drop_out),
        nn.ReLU()
    )# output_size = 32; RF = 5

    self.convblock3 = nn.Sequential(
        nn.Conv2d(out_size, out_size, kernel_size = (3,3), padding=1, dilation = 2, stride=2, bias=False),
        nn.BatchNorm2d(out_size),
        nn.Dropout(drop_out),
        nn.ReLU()
    )# output_size = 32; RF = 9

  def __call__(self, x):
    x = self.convblock1(x)
    x = self.convblock2(x) 
    x = self.convblock3(x) 
    
    return x

class DepthWiseConvolution(nn.Module):
  def __init__(self, input_size, output_size):
    super(DepthWiseConvolution, self).__init__()

    self.depthwise1 = nn.Sequential(
        nn.Conv2d(input_size, input_size, kernel_size = (3,3),padding= 1,groups = input_size),
        nn.ReLU()
    )
    self.pointwise1 =  nn.Sequential(
        nn.Conv2d(input_size, output_size, kernel_size = (1,1)),
        nn.BatchNorm2d(output_size),
        nn.ReLU()
    )
    self.depthwise2 = nn.Sequential(
        nn.Conv2d(output_size, output_size, kernel_size = (3,3),padding= 1,groups = output_size),
        nn.ReLU()
    )
    self.pointwise2 =  nn.Sequential(
        nn.Conv2d(output_size, output_size, kernel_size = (1,1)),
        nn.BatchNorm2d(output_size),
        nn.ReLU()
    )
    self.depthwise3 = nn.Sequential(
        nn.Conv2d(output_size, output_size, kernel_size = (3,3),padding= 1,groups = output_size),
        nn.ReLU()
    )
    
    self.pointwise3 =  nn.Sequential(
        nn.Conv2d(output_size, output_size, kernel_size = (1,1), padding= 0),
        nn.BatchNorm2d(output_size),
        nn.ReLU()
    )
   

  def __call__(self, x):
    x = self.depthwise1(x)
    x = self.pointwise1(x)
    x = self.depthwise2(x)
    x = self.pointwise2(x)    
    x = self.depthwise3(x)
    x = self.pointwise3(x)
    return x

# Block 1: 3, 5, 9    
# Block 2: 13, 17, 25
# Block 3: 25, 33, 41
# Block 4: 49, 57, 65

class Net_9(nn.Module):
  def __init__(self, drop_out = 0.1):
    super(Net, self).__init__()
    self.drop_out = drop_out

    # Input Block + Convolution Blocks
    self.layer1 = Block(3, 32, 0.1)
    self.layer2 = Block(32, 64, 0.1)

    # Depth-Wise Separable Convolutions
    self.layer3 = DepthWiseConvolution(64, 128)

   # OUTPUT BLOCK

    # output_size = 4; ; RF = 50
    self.gap = nn.Sequential(
        nn.AvgPool2d(kernel_size=7)
    ) # output_size = 1

    self.convblock5 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
    )# output_size = 1; ; RF = 28

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.gap(x)
    x = self.convblock5(x)
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1)