import cv2
from ultralytics import YOLO

# model = YOLO("best.pt")

# def detect_objects(frame):
#     results=model(frame)
#     detected_objects=[]
#     for r in results:
#         for box in r.boxes:
#             class_id=int(box.cls[0])
#             confidence=box.conf[0].item()

#             if confidence>0.8:
#                 label=model.names[class_id]
#                 detected_objects.append((label))
                
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                
#     return frame, detected_objects

# def main():
#     cap=cv2.VideoCapture(0)
    
#     while True:
#         ret, frame=cap.read()
#         if not ret:
#             break
        
#         processed_frame, detected_objects=detect_objects(frame)
        
#         cv2.imshow("Camera", processed_frame)
        
#         if detected_objects:
#             print("Detected objects:", detected_objects)
        
#         if cv2.waitKey(1) & 0xFF==ord('q'):
#             break


model=YOLO("best.pt")

results1 = model.predict(source="0",conf=0.5, iou=0.45, save=True,show=True)

print(results1)
