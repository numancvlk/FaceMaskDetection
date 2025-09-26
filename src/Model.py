#LIBRARIES
import torch
from torch import nn

class MaskModel(nn.Module):
    def __init__(self,
                 inputShape: int,
                 hiddenUnit: int,
                 outputShape: int):
        super().__init__()

        self.convLayer1 = nn.Sequential(
            nn.Conv2d(in_channels=inputShape,
                      out_channels=hiddenUnit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hiddenUnit),
            nn.ReLU(),
            nn.Conv2d(in_channels=hiddenUnit,
                      out_channels=hiddenUnit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hiddenUnit),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.convLayer2 = nn.Sequential(
            nn.Conv2d(in_channels=hiddenUnit,
                      out_channels=hiddenUnit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hiddenUnit),
            nn.ReLU(),
            nn.Conv2d(in_channels=hiddenUnit,
                      out_channels=hiddenUnit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hiddenUnit),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.convLayer3 = nn.Sequential(
            nn.Conv2d(in_channels=hiddenUnit,
                      out_channels=hiddenUnit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hiddenUnit),
            nn.ReLU(),
            nn.Conv2d(in_channels=hiddenUnit,
                      out_channels=hiddenUnit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hiddenUnit),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hiddenUnit * 28 * 28,
                      out_features=outputShape)
        )
    
    def forward(self,x):
        x = self.convLayer1(x)
        x = self.convLayer2(x)
        x = self.convLayer3(x)
        x = self.classifier(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)