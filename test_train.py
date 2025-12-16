import cv2
from ultralytics import YOLO
from PIL import Image

model = YOLO("yolo11n.yaml")

model=YOLO("yolo11n.pt")

model=YOLO("yolo11n.yaml").load("yolo11n.pt ")
if __name__ == '__main__':
    results=model.train(data="coco128.yaml", epochs=100, imgsz=640, save=True)
    
