import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def prepare_paths(video_name: str, campus_number: str):
    """
    Проверяет пути к видео и папке.
    Если папка не найдена, создаст её.
    Возвращает пути к видео и папке.
    """

    video_path = os.path.join(DATA_DIR, "raw", f"{video_name}.mp4")
    output_path = os.path.join(DATA_DIR, "processed", campus_number)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Видео '{video_path}' не найдено! ")
    os.makedirs(output_path, exist_ok=True)

    return video_path, output_path
