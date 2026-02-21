import cv2
import os


def extract_frames(video_path, output_path, frame_step: int = 3):
    """
    Извлекает кадры из видео и сохраняет их в указанную папку.
    Возвращает количество сохраненных кадров.
    """

    cap = cv2.VideoCapture(video_path)

    frames_count = frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_step == 0:
            frame_filename = os.path.join(
                output_path,
                f"{frames_count + 1:04d}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            frames_count += 1

        frame_index += 1

    cap.release()
    return frames_count
