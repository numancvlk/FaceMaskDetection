#LIBRARIES
import os  # Dosya ve klasör işlemleri için
import xml.etree.ElementTree as ET  # Dosya ve klasör işlemleri için
from shutil import copy2  # Dosyaları bir yerden başka bir yere kopyalamak için

annotionsDir = "dataset\\archive\\annotations" # XML dosyalarının olduğu klasör
imagesDir = "dataset\\archive\\images" # Tüm resimlerin olduğu klasör
outputDir = "sortedDataset" # Resimleri ayıracağımız ve kaydedeceğimiz klasör

classes = ["with_mask", "without_mask","mask_weared_incorrect"]  # Sınıf isimleri

for cls in classes:
    os.makedirs(os.path.join(outputDir,cls), exist_ok=True) # Her sınıf için klasör oluştur

for xmlFile in os.listdir(annotionsDir): # annotations klasöründeki tüm XML dosyalarını sırayla oku
    tree = ET.parse(os.path.join(annotionsDir,xmlFile)) # XML dosyasını aç
    root = tree.getroot() # XML’in kök elementini al
    filename = root.find("filename").text # XML içinde hangi resimle ilişkili olduğunu bul
    label = root.find("object").find("name").text

    src = os.path.join(imagesDir, filename) #Kaynak resmin yolu
    dst = os.path.join(outputDir,label,filename) # Hedef klasör ve dosya yolu

    if os.path.exists(src):
        copy2(src,dst) # Resmi ilgili klasöre kopyala

