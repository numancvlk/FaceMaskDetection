#LIBRARIES
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision import datasets

import matplotlib.pyplot as plt

BATCH_SIZE = 64


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),  # %50 ihtimalle yatay çevir
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # renk değişikliği
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

trainDataset = datasets.ImageFolder("datasets\Face Mask Dataset\Train", transform=transform)
testDataset = datasets.ImageFolder("datasets\Face Mask Dataset\Test", transform=transform)


trainDataLoader = DataLoader(dataset=trainDataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

testDataLoader = DataLoader(dataset=testDataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False)


image, label = trainDataset[1]

plt.title(trainDataset.classes[label])
plt.imshow(image.permute(1,2,0))
plt.show()
print(image.shape)
