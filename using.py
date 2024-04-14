import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

model = YOLO("yolov8n")
cap = cv2.VideoCapture(0)

while True:

    status: bool
    frame: np.ndarray
    status, frame = cap.read()
    frame: np.ndarray = cv2.resize(frame, (1024, 1024))
    key: int = cv2.waitKey(delay=1)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not status or key == ord("q") or key == 27:
        break

    predict: list[Results] = model.predict(
        source=frame,
        device=device,
    )
    cv2.imshow(
        winname="Camera",
        mat=predict[0].plot()
    )
