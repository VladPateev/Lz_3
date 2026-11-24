import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from model import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_FILE,
    TRAIN_DIR,
    AnimalNet,
)
from pillow_processing import AnimalDataset


def main():
    torch.manual_seed(42)
    if DEVICE.type == "cpu":
        torch.set_num_threads(min(6, torch.get_num_threads()))
    print(f"Обучаем на {DEVICE}")
    print(f"Эпох: {EPOCHS}, Размер батча: {BATCH_SIZE}")

    #загружаем датасет
    print("Загрузка картинок через Pillow...")
    full_dataset = AnimalDataset(TRAIN_DIR)

    #делим на обучение и тесты
    train_count = int(0.8 * len(full_dataset))
    val_count = len(full_dataset) - train_count
    train_data, val_data = random_split(full_dataset, [train_count, val_count])

    #делаем батчинг (по 64 картинки)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    model = AnimalNet().to(DEVICE)

    # Функция потерь (CrossEntropy) и оптимизатор (Adam)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_accuracy = 0.0

    print("\nНачинаем обучение...\n")

    #главный цикл по эпохам
    for epoch in range(1, EPOCHS + 1):
        # --- ОБУЧЕНИЕ ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            #отзеркаливание для лучшего узнавания
            if torch.rand(1).item() < 0.5:
                images = images.flip(-1)

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            #обнуляем градиенты
            optimizer.zero_grad()

            #проход
            outputs = model(images)

            #считаем ошибки
            loss = criterion(outputs, labels)

            #проход обранто
            loss.backward()

            #новые веса
            optimizer.step()

            #подсчет точности
            train_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        epoch_loss = train_loss / train_total
        epoch_acc = 100.0 * train_correct / train_total

        # --- ПРОВЕРКА  ---
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # отключаем расчет градиентов
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100.0 * val_correct / val_total

        print(f"Эпоха [{epoch:>2}/{EPOCHS}] | Потеря: {epoch_loss:.4f} | Обучение: {epoch_acc:.1f}% | тесты: {val_acc:.1f}%")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), MODEL_FILE)
            print(" Новая модель сохранена!")
    print("-" * 50)
    print(f"Обучение завершено! Лучшая точность: {best_accuracy:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
