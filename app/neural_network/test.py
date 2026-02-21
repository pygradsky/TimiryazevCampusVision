import torch
from PIL import Image
from model import SimpleCNN
from app.utils.transforms import data_transform
from app.config import SAVED_MODELS_DIR, TEST_IMAGES_PATH

# -------------------------------
# Настройка устройства
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Загружаем модель
# -------------------------------
model = SimpleCNN()
model_name = input("Введите имя сохраненной модели:\n>>> ")
model_path = SAVED_MODELS_DIR / model_name
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
except FileNotFoundError:
    print(f"Путь {model_path} не был найден.")

# -------------------------------
# Предсказание для всех изображений в TEST_IMAGES_PATH
# -------------------------------
for img_file in TEST_IMAGES_PATH.glob("*.jpg"):
    img = Image.open(img_file).convert('RGB')
    img_tensor = data_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    print(f"{img_file.name}: Предсказанный класс -> {predicted.item() + 1}")
