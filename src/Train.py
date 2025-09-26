#SCRIPTS
from Helpers import accuracy,printTrainTime,trainStep,testStep,modelSummary
from Model import MaskModel, device
from Dataset import trainDataLoader,testDataLoader

#LIBRARIES
import torch
from torch import nn

import random
from timeit import default_timer
from tqdm.auto import tqdm

random.seed(32)

LEARNING_RATE = 0.001

myMaskModel = MaskModel(inputShape=3,
                        hiddenUnit=128,
                        outputShape=3).to(device) #OUTPUT SHAPE CLASS SAYISI!!!

lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=myMaskModel.parameters(),
                            lr=LEARNING_RATE)

epochs = 10

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
    