from app.utils.frames_loader import extract_frames
from app.utils.prepare_paths import prepare_paths


def safe_extract(video_name, campus_number):
    try:
        video_path, output_path = prepare_paths(video_name, campus_number)
        frames_count = extract_frames(video_path, output_path)

        return f"Извлечено кадров: {frames_count}\nФотографии были сохранены в '{output_path}'"

    except ValueError as e:
        return f"Ошибка параметров: {e}"

    except FileNotFoundError as e:
        return f"Ошибка пути: {e}"

    except Exception as e:
        return f"Неизвестная ошибка: {e}"
