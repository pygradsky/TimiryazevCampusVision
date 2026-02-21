def validate_frame_step(frame_step: int):
    if frame_step <= 0:
        raise ValueError("Шаг должен быть больше нуля!")
