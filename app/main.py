import torch
import torch.nn as nn
import torch.optim as optim
from app.neural_network.model import SimpleCNN
from app.utils.data_utils import get_loaders
import os
from pathlib import Path

# -------------------------------
# Настройка устройства
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Получаем DataLoader'ы
# -------------------------------
train_loader, test_loader, classes = get_loaders()
print("Классы:", classes)
print("Размер обучающей выборки:", len(train_loader.dataset))
print("Размер тестовой выборки:", len(test_loader.dataset))

# -------------------------------
# Создаем модель
# -------------------------------
model = SimpleCNN(num_classes=len(classes))
model.to(device)

# -------------------------------
# Критерий и оптимизатор
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# Обучение
# -------------------------------
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Эпоха {epoch + 1}, loss: {running_loss / len(train_loader):.4f}")

print("Обучение завершено!")

# -------------------------------
# Сохраняем модель
# -------------------------------
save_dir = Path(__file__).parent / "saved_models"
os.makedirs(save_dir, exist_ok=True)  # создаём папку, если её нет
save_path = save_dir / "simple_cnn.pth"

torch.save(model.state_dict(), save_path)
print(f"Модель успешно сохранена в {save_path}")

# -------------------------------
# Тестирование
# -------------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Точность на тесте: {100 * correct / total:.2f}%")
