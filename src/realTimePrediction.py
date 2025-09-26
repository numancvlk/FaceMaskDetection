#LIBRARIES
import torch
import cv2 as cv

from torchvision import transforms
from PretrainedModel import getMaskModel,device

from collections import deque

from SplitData import classes

from facenet_pytorch import MTCNN #YÜZÜ ALGILAMAK İÇİN


model = getMaskModel(numClasses=3,device=device)
model.eval()

capture = cv.VideoCapture(0)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),  # Modelin eğitimde kullandığı boyuta resize et
    transforms.ToTensor(), # Görüntüyü PyTorch tensörüne dönüştür (C x H x W formatı)
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]) 
])

mtcnn = MTCNN(keep_all=True, device=device)
predictionQueue = deque(maxlen=15)

while True:
    isTrue, frame = capture.read()

    if not isTrue:
        break

    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Yüzleri tespit et
    boxes, _ = mtcnn.detect(img_rgb)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            roi = img_rgb[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue

            input_tensor = transform(roi).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                pred_class = torch.argmax(output, dim=1).item()
                pred_label = classes[pred_class]
                predictionQueue.append(pred_class)
        
        if predictionQueue:
            stable_pred = max(set(predictionQueue), key=predictionQueue.count)
            pred_label = classes[stable_pred]

            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, pred_label, (x1, y1-10),cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv.imshow("Mask Detection", frame)

    if cv.waitKey(1) == 27:
        break

capture.release()
cv.destroyAllWindows()