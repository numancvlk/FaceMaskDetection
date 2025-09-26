#SCRIPTS
from PretrainedModel import getMaskModel,device
from Dataset import testDataset,trainDataset

#LIBRARIES
import torch
import random
import matplotlib.pyplot as plt

newModel = getMaskModel(numClasses=2,device=device)

def makePredictions(model:torch.nn.Module,
                    data:list,
                    device:torch.device=device):

  predProbs = []
  model.to(device)
  model.eval()

  with torch.inference_mode():
    for sample in data:
      sample = torch.unsqueeze(sample,dim=0).to(device)

      predLogits = model(sample)

      predProb = torch.softmax(predLogits.squeeze(), dim=0)

      predProbs.append(predProb.cpu())

  return torch.stack(predProbs)



random.seed(41)
testSamples = []
testLabels = []

for sample, label in random.sample(list(testDataset), k=9):
  testSamples.append(sample)
  testLabels.append(label)


prediction = makePredictions(model=newModel,
                             data=testSamples,
                             device=device)

predictionClasses = prediction.argmax(dim=1)


# PLOT PREDICTIONS WITH COLORS
plt.figure(figsize=(9,9))
nrows = 3
ncols = 3

for i, sample in enumerate(testSamples):
    plt.subplot(nrows, ncols, i+1)

    # Görüntüyü göster
    plt.imshow(sample.permute(1,2,0))
    plt.axis('off')  # Eksenleri gizle

    # Tahmin ve gerçek sınıf isimleri
    predLabel = testDataset.classes[predictionClasses[i]]  # modelin tahmini
    trueLabel = testDataset.classes[testLabels[i]]         # gerçek sınıf

    # Renk belirle
    color = "green" if predLabel == trueLabel else "red"

    # Başlık ekle
    plt.title(f"P: {predLabel}\nT: {trueLabel}", color=color, fontsize=10)

plt.tight_layout()
plt.show()

for idx, ch in enumerate(trainDataset.classes):
    print(idx, ch)