from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from app.neural_network.model import SimpleCNN

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "saved_models"


# Путь к новой картинке
img_path = r"C:\Users\Fedor\PycharmProjects\TimiryazevCampusVision\images.jpg"

# Параметры модели
num_classes = 3
img_size = 64  # должен совпадать с размером, который использовался при обучении

# Трансформации (те же, что и для обучения)
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# Загружаем модель
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_DIR / "simple_cnn.pth", map_location=device))
model.to(device)
model.eval()

# Загружаем изображение
img = Image.open(img_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)  # добавляем batch dimension

# Предсказание
with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)

# Классы (тоже нужно взять из train)
classes = ['1','2','3']  # замени на dataset.classes, если нужно
print(f"Предсказанный класс: {classes[predicted.item()]}")
