#LIBRARIES
import torch

from PretrainedModel import device

def accuracy(yTrue,yPred):
    correct = torch.eq(yTrue,yPred).sum().item()
    acc = correct / len(yTrue)
    return acc

def printTrainTime(start,end,device):
    totalTime = end - start
    print(f"Total train time is {totalTime} on the {device}")

def trainStep(model: torch.nn.Module,
              dataLoader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              lossFn: torch.nn.Module,
              accFn,
              device: torch.device = device):

    trainLoss, trainAccuracy = 0,0
    model.train()

    for batch, (xTrain,yTrain) in enumerate(dataLoader):
        xTrain, yTrain = xTrain.to(device), yTrain.to(device)

        trainPred = model(xTrain)

        loss = lossFn(trainPred,yTrain)
        trainLoss += loss

        acc = accFn(yTrue = yTrain, yPred = trainPred.argmax(dim=1))
        trainAccuracy += acc

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if batch % 150 == 0:
            print(f"Looked at {batch * len(xTrain)} / {len(dataLoader.dataset)} samples")
    
    trainLoss /= len(dataLoader)
    trainAccuracy /= len(dataLoader)
    print(f"TRAIN LOSS = {trainLoss:.5f} | TRAIN ACCURACY = {trainAccuracy:.5f}%")


def testStep(model: torch.nn.Module,
             dataLoader: torch.utils.data.DataLoader,
             lossFn: torch.nn.Module,
             accFn,
             device: torch.device = device):
    
    testLoss,testAccuracy = 0,0
    model.eval()

    with torch.inference_mode():
        for xTest, yTest in dataLoader:
            xTest,yTest = xTest.to(device), yTest.to(device)

            testPred = model(xTest)

            loss = lossFn(testPred, yTest)
            testLoss += loss

            acc = accFn(yTrue = yTest, yPred = testPred.argmax(dim=1))
            testAccuracy += acc

        testLoss /= len(dataLoader)
        testAccuracy /= len(dataLoader)
        print(f"TEST LOSS = {testLoss:.5f} | TEST ACCURACY = {testAccuracy:.5f}%")

def modelSummary(model:torch.nn.Module,
                 dataLoader: torch.utils.data.DataLoader,
                 lossFn: torch.nn.Module,
                 accFn,
                 device: torch.device = device):
    summaryLoss,summartAccuracy = 0,0
    model.eval()

    with torch.inference_mode():
        for xTest, yTest in dataLoader:
            xTest,yTest = xTest.to(device), yTest.to(device)

            summaryPred = model(xTest)

            loss = lossFn(summaryPred, yTest)
            summaryLoss += loss

            acc = accFn(yTrue = yTest, yPred = summaryPred.argmax(dim=1))
            summartAccuracy += acc

        summaryLoss /= len(dataLoader)
        summartAccuracy /= len(dataLoader)
        return {"MODEL NAME": model.__class__.__name__,
                "MODEL LOSS": summaryLoss,
                "MODEL ACCURACY": summartAccuracy}