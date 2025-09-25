#LIBRARIES
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision import datasets

import matplotlib.pyplot as plt

BATCH_SIZE = 128


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

allDatas = datasets.ImageFolder("sortedDataset",transform=transform) #TÜM DATALARI ALMAK


trainSize = int(0.8 * len(allDatas)) #%80 train verisi
testSize = len(allDatas) - trainSize #%20 test verisi

#Verileri verdiğimiz oranlarda trainDataset ve testDataset üzerinde dağıtıyor
trainDataset, testDataset = random_split(allDatas, [trainSize,testSize])

trainDataLoader = DataLoader(dataset=trainDataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

testDataLoader = DataLoader(dataset=testDataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False)


image, label = trainDataset[1]

plt.title(allDatas.classes[label])
plt.imshow(image.permute(1,2,0))
plt.show()
