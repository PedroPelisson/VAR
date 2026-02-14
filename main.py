import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video_path = "lance.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:

        results = model.track(frame, persist=True, classes=[0], device="cpu")
        annotated_frame = results[0].plot()
        cv2.imshow("VAR", annotated_frame)

        if cv2.waitKey(1) == 27:
            break
    
    else:
        break

cap.release()
cv2.destroyAllWindows()