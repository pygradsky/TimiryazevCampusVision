import cv2
import os

BASE_DIR = r"C:\Users\Professional\PycharmProjects\TimiryazevCampusVision"


def video_to_frames(video_name: str, campus_name: str):
    """Извлекает кадры из видео и сохраняет их в папке, соответствующей выбранному корпусу."""

    video_path = os.path.join(BASE_DIR, "videos", f"{video_name}.mp4")
    output_path = os.path.join(BASE_DIR, "campuses", campus_name)

    if not os.path.exists(video_path):
        return f"Видео '{video_path}' не найдено."
    if not os.path.exists(output_path):
        return f"Папка '{output_path}' не найдена."

    print("\nПодождите, идет извлечение кадров из видео...")
    cap = cv2.VideoCapture(video_path)
    frames_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_path, f"frame_{frames_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frames_count += 1

    cap.release()
    print(f"Было извлечено кадров: {frames_count}")
    return None


def main():
    while True:
        print("\n1. Нарезать видео на кадры\n2. Выйти")
        choice = input(">>> ")
        if choice == "2":
            print("Выход.")
            break
        elif choice == "1":
            video_name = input("Введите название видео (без расширения): ")
            campus_name = input("Введите название корпуса: ")
            video_to_frames(video_name, campus_name)
        else:
            print("Некорректный выбор. Попробуйте снова.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Досрочное завершение програмы")
