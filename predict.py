import os
import random
import sys

import torch

from model import CLASS_NAMES, DEVICE, MODEL_FILE, TEST_DIR, AnimalNet
from pillow_processing import load_image


def predict(image_path):
    """Предсказывает животное на одной конкретной картинке"""
    #загружаем веса
    model = AnimalNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()

    #делаем картинку
    image_tensor = load_image(image_path)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    #обучение
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    best_index = probabilities.argmax().item()
    best_animal = CLASS_NAMES[best_index]
    confidence = probabilities[best_index].item() * 100.0

    print(f"\nКартинка: {os.path.basename(image_path)}")
    print(f"Ответ: {best_animal} ({confidence:.1f}%)")
    print("Вероятности по всем классам:")
    for i, name in enumerate(CLASS_NAMES):
        pct = probabilities[i].item() * 100.0
        print(f"  {name:12s}: {pct:5.1f}%")


def main():
    if not os.path.exists(MODEL_FILE):
        print(f"Ошибка: веса {MODEL_FILE} не найдены. Сначала запустите train.py!")
        return

    #если путь к картинке передан при запуске
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        if os.path.exists(img_path):
            predict(img_path)
        else:
            print(f"Файл {img_path} не найден!")
        return

    #иначе проверяет 5 случайных картинок из папки test
    if os.path.exists(TEST_DIR):
        files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        chosen = random.sample(files, min(5, len(files)))

        print("Проверка на 5 случайных картинках из test/...")
        for f in chosen:
            predict(os.path.join(TEST_DIR, f))
            print("-" * 40)
    else:
        print(f"Папка {TEST_DIR} не найдена. Укажи путь к файлу: python predict.py <файл.jpg>")


if __name__ == "__main__":
    main()
