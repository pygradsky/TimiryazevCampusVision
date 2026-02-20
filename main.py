import torch
import torch.nn as nn
import torch.optim as optim

# Импортируем нашу утилиту для загрузки данных
from app.utils.image_dataloader import get_image_dataloader


# ============================================================
# ОПРЕДЕЛЕНИЕ НЕЙРОСЕТИ
# ============================================================
class SimpleCNN(nn.Module):
    """
    Простая сверточная нейросеть для классификации изображений.
    Архитектура:
    - 2 сверточных слоя для извлечения признаков из изображений
    - 2 полносвязных слоя для классификации
    """

    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        # Первый сверточный блок:
        # Conv2d: 3 входных канала (RGB), 16 выходных, ядро 3x3
        # ReLU: функция активации (убирает отрицательные значения)
        # MaxPool2d: уменьшает размер изображения в 2 раза
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 224x224 -> 112x112
        )

        # Второй сверточный блок:
        # Увеличиваем количество фильтров с 16 до 32
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 112x112 -> 56x56
        )

        # Полносвязные слои для классификации:
        # После сверток у нас изображение 56x56 с 32 каналами = 32*56*56 признаков
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # Сжимаем до 128 признаков
        self.fc2 = nn.Linear(128, num_classes)   # Выход: количество классов

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Прямой проход через сеть,
        x - входное изображение размером [batch, 3, 224, 224]
        """

        # Пропускаем через сверточные слои
        x = self.conv1(x)
        x = self.conv2(x)

        # "Разворачиваем" многомерный тензор в одномерный вектор
        x = x.view(x.size(0), -1)

        # Пропускаем через полносвязные слои
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# ============================================================
# ФУНКЦИЯ ОБУЧЕНИЯ
# ============================================================
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
    """
    Обучение модели.

    Параметры:
    - model: нейросеть
    - train_loader: загрузчик данных
    - criterion: функция потерь (loss)
    - optimizer: оптимизатор (обновляет веса сети)
    - device: устройство (CPU или GPU)
    - num_epochs: количество эпох обучения
    """

    model.train()  # Переводим модель в режим обучения

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Проходим по всем батчам данных
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Переносим данные на нужное устройство (CPU/GPU)
            images = images.to(device)
            labels = labels.to(device)

            # 1. Обнуляем градиенты (важно делать каждую итерацию!)
            optimizer.zero_grad()

            # 2. Прямой проход: получаем предсказания модели
            outputs = model(images)

            # 3. Вычисляем ошибку (loss)
            loss = criterion(outputs, labels)

            # 4. Обратное распространение ошибки (backpropagation)
            loss.backward()

            # 5. Обновляем веса сети
            optimizer.step()

            # Статистика
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Выводим статистику после каждой эпохи
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Эпоха [{epoch + 1}/{num_epochs}] | Loss: {epoch_loss:.4f} | Точность: {epoch_acc:.2f}%")


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================
def main():
    # Определяем устройство: GPU если доступен, иначе CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Загружаем данные с помощью нашей утилиты
    print("Загрузка данных...")
    dataset, train_loader = get_image_dataloader()

    # Получаем количество классов (папок с изображениями)
    num_classes = len(dataset.classes)
    print(f"Найдено классов: {num_classes}")
    print(f"Классы: {dataset.classes}")
    print(f"Всего изображений: {len(dataset)}")

    # Создаем модель и переносим на устройство
    model = SimpleCNN(num_classes=num_classes).to(device)
    print(f"\nМодель создана: {model.__class__.__name__}")

    # Функция потерь: CrossEntropyLoss для классификации
    criterion = nn.CrossEntropyLoss()

    # Оптимизатор Adam: обновляет веса сети
    # lr (learning rate) - скорость обучения
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Запускаем обучение
    print("\nНачинаем обучение...")
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=5)

    # Сохраняем обученную модель
    torch.save(model.state_dict(), "campus_classifier.pth")
    print("\nМодель сохранена в 'campus_classifier.pth'")


if __name__ == "__main__":
    main()
