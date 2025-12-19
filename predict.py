from ultralytics import YOLO
import os
from PIL import Image


model=YOLO("best.pt")

print(model.names)

im1 = Image.open("predict0.jpg")
im2 = Image.open("predict1.jpg")
im3 = Image.open("predict2.jpg")
im4 = Image.open("test_none.jpg")
results1 = model.predict(source=im1,conf=0.5, iou=0.45, save=True)

results2 = model.predict(source=im2,conf=0.5, iou=0.45, save=True)

results3 = model.predict(source=im3,conf=0.5, iou=0.45, save=True)

results4 = model.predict(source=im4,conf=0.5, iou=0.45, save=True)

# from ultralytics import YOLO

# metrics = model.val(data="data.yaml", imgsz=640, batch=16, conf=0.25, iou=0.65, save=True)