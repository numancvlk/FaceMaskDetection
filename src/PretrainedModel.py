#LIBRARIES
import torch
from torch import nn
from torchvision import models

def getMaskModel(numClasses = 3,device="cpu"):
    # MobileNetV2 modelini önceden eğitilmiş ağırlıklarla yükle
    # pretrained=True → ImageNet üzerinde eğitilmiş ağırlıkları alır
    model = models.mobilenet_v2(pretrained = True) 

    # Son sınıflandırma katmanını değiştir
    # model.classifier[1] genellikle Linear katmanı içerir
    model.classifier[1] = nn.Linear(model.last_channel,numClasses)
    model = model.to(device)
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)