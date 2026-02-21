import torch
import torch.nn as nn
import torch.optim as optim

from model import SimpleCNN
from app.utils.data_utils import get_loaders
from app.config import SAVED_MODELS_DIR

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
model = SimpleCNN().to(device)

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
    running_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"{epoch + 1}. Потери: {running_loss / len(train_loader):.4f}")

print("Обучение завершено!")

# -------------------------------
# Сохраняем модель
# -------------------------------
model_name = input("Введите название для сохраняемой модели:\n>>> ")
if not model_name:
    pass
else:
    try:
        save_path = SAVED_MODELS_DIR / f"{model_name}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Модель успешно сохранена в {save_path}")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")

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
        correct += int((predicted == labels).sum().item())
        total += labels.size(0)

print(f"Точность на тесте: {100 * correct / total:.2f}%")
