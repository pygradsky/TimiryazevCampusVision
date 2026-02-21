import os
import sys

# Путь к утилите
from app.utils.safe_extract import safe_extract

# Пути к скриптам
from app.config import TRAIN_MODEL_SCRIPT
from app.config import TEST_MODEL_SCRIPT

# Пути к данным
from app.config import VIDEOS_DIR
from app.config import CLASSES_DIR


def main():
    while True:
        print("\n=== TimiryazevCampusVision Menu ===")
        print("1. Нарезать кадры с видео")
        print("2. Обучить модель")
        print("3. Предсказать на новых изображениях")
        print("4. Выйти")
        choice = input("\nВыберите действие (1-4):\n>>> ")

        if choice == "1":
            video_name = input("Введите имя видео (без расширения): ")
            corpus_number = input("Введите номер корпуса: ")

            video_path = os.path.join(VIDEOS_DIR, video_name)
            output_path = os.path.join(CLASSES_DIR, corpus_number)

            result = safe_extract(video_path, output_path)
            print(result)

        elif choice == "2":
            print("Запуск обучения модели...")
            os.system(f'python "{TRAIN_MODEL_SCRIPT}"')
        elif choice == "3":
            print("Запуск предсказания на новых изображениях...")
            os.system(f'python "{TEST_MODEL_SCRIPT}"')
        elif choice == "4":
            print("Выход...")
            sys.exit(0)
        else:
            print("Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()
