from pathlib import Path

# Базовая папка проекта
BASE_DIR = Path(__file__).parent.parent

# Приложение
APP_DIR = BASE_DIR / "app"

# Пути к данным
DATA_DIR = BASE_DIR / "data"
CLASSES_DIR = TRAIN_DIR = TEST_DIR = DATA_DIR / "processed"
VIDEOS_DIR = DATA_DIR / "raw"

# Сохранения моделей и с ними связанного
BASE_MODEL_DIR = APP_DIR / "neural_network"
SAVED_MODELS_DIR = APP_DIR / "saved_models"

TRAIN_MODEL_SCRIPT = BASE_MODEL_DIR / "train.py"
TEST_MODEL_SCRIPT = BASE_MODEL_DIR / "test.py"

# Путь к новой картинке для предсказания (пример)
TEST_IMAGES_PATH = BASE_DIR / "test_images"

# Параметры модели/данных
IMG_SIZE = 64
BATCH_SIZE = 32
NUM_CLASSES = 3
