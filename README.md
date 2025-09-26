# Face Mask Detection

# [TR]
## Projenin Amacı
Bu proje, insanların maske takıp takmadığını anlık olarak kamera üzerinde gösteren bir uygulamadır.

## 💻 Kullanılan Teknolojiler
- Python 3.11.8
- PyTorch: Model oluşturma, eğitim ve tahmin için.
- Torchvision: Dataset ve transform işlemleri için.
- OpenCV: Kamera görüntüsü yakalamak için
- Matplotlib: Eğitim verilerini ve tahmin sonuçlarını görselleştirmek için.
- TQDM: Eğitim sürecinde ilerleme çubuğu göstermek için.
- Timeit: Eğitim süresini ölçmek için.
- Facenet-PyTorch (MTCNN): Yüz tespiti
- MobileNetV2: Maskeli / maskesiz yüz sınıflandırma

## ⚙️ Kurulum
GEREKLİ KÜTÜPHANELERİ KURUN
```bash
pip install torch torchvision matplotlib opencv-python tqdm mtcnn facenet-pytorch
```

## 📂 Kullanılan Dataset
https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset

## 🚀 Çalıştırma
1. Önce veriler için **Dataset.py** dosyasını çalıştırın.
2. Modeli oluşturmak için **Model.py** dosyasını çalıştırın.
3. Modelin ihtiyaç duyduğu fonksiyonlar için **Helpers.py** dosyasını çalıştırın.
4. Modeli eğitmek için **Train.py** dosyasını çalıştırın.
5. Modelin tahminleri için **Predict.py** dosyasını çalıştırın.
6. Modelin kamera üzerinden tahminler yapmasını sağlamak için **realTimePrediction.py** dosyasını çalıştırın.

## BU PROJE HİÇBİR ŞEKİLDE TİCARİ AMAÇ İÇERMEMEKTEDİR.


# [EN]
## Project Purpose
This project is an application that shows in real time on the camera whether people are wearing a mask or not.

## 💻 Technologies Used
- Python 3.11.8
- PyTorch: For creating, training, and predicting the model.
- Torchvision: For dataset and transform operations.
- OpenCV: To capture camera images.
- Matplotlib: To visualize training data and prediction results.
- TQDM: To show a progress bar during training.
- Timeit: To measure training time.
- Facenet-PyTorch (MTCNN): Face detection
- MobileNetV2: Masked / unmasked face classification

## ⚙️ Installation
INSTALL REQUIRED LIBRARIES
```bash
pip install torch torchvision matplotlib opencv-python tqdm mtcnn facenet-pytorch
```

## 📂 Used Dataset
https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset

## 🚀 Run

1. First run Dataset.py for the data.
2. Run Model.py to create the model.
3. Run Helpers.py for the functions required by the model.
4. Run Train.py to train the model.
5. Run Predict.py for the model predictions.
6. Run realTimePrediction.py to enable the model to make predictions via the camera.

## THIS PROJECT DOES NOT CONTAIN ANY COMMERCIAL PURPOSE IN ANY WAY.
