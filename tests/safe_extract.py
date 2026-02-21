from app.utils.frames_loader import extract_frames
from app.utils.prepare_paths import prepare_paths
from app.utils.validate_frame_step import validate_frame_step


def safe_extract(video_name, campus_number):
    try:
        custom_frame_step = int(input("Введите шаг нарезки (>0): "))
        video_path, output_path = prepare_paths(video_name, campus_number)
        validate_frame_step(custom_frame_step)

        print("\nПодождите, идет извлечение кадров из видео...")
        frames_count = extract_frames(video_path, output_path, custom_frame_step)

        print(f"Было извлечено кадров: {frames_count}")

    except ValueError as e:
        print("Ошибка параметров:", e)

    except FileNotFoundError as e:
        print("Ошибка пути:", e)

    except Exception as e:
        print("Неизвестная ошибка:", e)
