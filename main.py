from ultralytics import YOLO
import cv2

model = YOLO('yolov8s-pose.pt')
cap = cv2.VideoCapture('lance2.mp4')
tracker_config = "bytetrack.yaml" 

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model.track(
        frame, 
        persist=True, 
        tracker=tracker_config,
        conf=0.1,
        iou=0.8,
        imgsz=960,
        verbose=False
    )

    if results[0].boxes.id is not None:
        annotated_frame = results[0].plot()
        ids = results[0].boxes.id.cpu().numpy()
        cv2.imshow("VAR", annotated_frame)
    
    if cv2.waitKey(5) == 27:
        break

cap.release()
cv2.destroyAllWindows()