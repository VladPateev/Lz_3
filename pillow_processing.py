import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from model import CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH


#обработка картинок
def load_image(image_path):
    #Открываем и переводим в RGB
    img = Image.open(image_path).convert("RGB")

    # Приводим все картинки к одному размеру
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))

    #делим на 255 для получения значений от 0 до 1
    pixels = np.array(img, dtype=np.float32) / 255.0

    # Ставим каналы RGB перед высотой и шириной
    return torch.from_numpy(pixels).permute(2, 0, 1)

#подает всесь датасет для обучения
class AnimalDataset(Dataset):
    def __init__(self, train_folder):
        self.files = []
        self.labels = []

        #обходим папки с каждым животным
        for label_index, animal in enumerate(CLASS_NAMES):
            animal_folder = os.path.join(train_folder, f"{animal}_train")
            if not os.path.exists(animal_folder):
                continue

            for file_name in os.listdir(animal_folder):
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(animal_folder, file_name)
                    self.files.append(full_path)
                    self.labels.append(label_index)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        #загружаем картинку через Pillow
        image_tensor = load_image(file_path)
        return image_tensor, label
