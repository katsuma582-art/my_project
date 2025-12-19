import cv2
from ultralytics import YOLO
from PIL import Image
import os

if __name__ == "__main__":
    model = YOLO("yolo11x.pt")
    results = model.train(data="./data.yaml", epochs=50, imgsz=640, save=True)