import os

import torch
from torch import nn

#Размеры картинок
IMG_WIDTH = 144
IMG_HEIGHT = 96
CHANNELS = 3

#входы
INPUT_SIZE = IMG_WIDTH * IMG_HEIGHT * CHANNELS

#Список классов
CLASS_NAMES = ["слон", "бабочка", "коровы", "овца", "белка"]
NUM_CLASSES = len(CLASS_NAMES)  # 5 классов

#Настройки обучения
BATCH_SIZE = 64        
EPOCHS = 8
LEARNING_RATE = 0.001  


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#пути к данным
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "animal-dataset", "versions", "1", "animal_dataset_intermediate")
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
TEST_DIR = os.path.join(DATASET_PATH, "test")
MODEL_FILE = os.path.join(BASE_DIR, "best_model_80.pth")

#архитектура слоев
class AnimalNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            # Блок 1: 96x64 -> 48x32
            nn.Conv2d(CHANNELS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Блок 2: 48x32 -> 24x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Блок 3: 24x16 -> 12x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Блок 4: 12x8 -> 6x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Приводим к фиксированному размеру 4x4 независимо от пропорций картинки
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
