#SCRIPTS
from Helpers import accuracy,printTrainTime,trainStep,testStep,modelSummary
from PretrainedModel import getMaskModel,device
from Dataset import trainDataLoader,testDataLoader

#LIBRARIES
import torch
from torch import nn

import random
from timeit import default_timer
from tqdm.auto import tqdm

random.seed(32)

LEARNING_RATE = 0.0001

myMaskModel = getMaskModel(numClasses=3,device=device)

for param in myMaskModel.features.parameters():
    param.requires_grad = False

for block in myMaskModel.features[-2:]:
    for param in block.parameters():
        param.requires_grad = True

lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=myMaskModel.classifier.parameters(),
                            lr=LEARNING_RATE)

epochs = 30

startTrainTimer = default_timer()

for epoch in tqdm(range(epochs)):
    print(f"EPOCH = {epoch}\n-----")

    trainStep(model=myMaskModel,
              dataLoader=trainDataLoader,
              optimizer=optimizer,
              lossFn=lossFn,
              accFn=accuracy,
              device=device
              )
    
    testStep(model=myMaskModel,
             dataLoader=testDataLoader,
             lossFn=lossFn,
             accFn=accuracy,
             device=device)
    
endTrainTimer = default_timer()

printTrainTime(start=startTrainTimer,
               end=endTrainTimer,
               device=device)

modelSum = modelSummary(model=myMaskModel,
                        dataLoader=testDataLoader,
                        lossFn=lossFn,
                        accFn=accuracy,
                        device=device)

print(modelSum)

torch.save(myMaskModel.state_dict(),"myMaskModel.pth")
print("AĞIRLIKLAR KAYDEDİLDİ")
    