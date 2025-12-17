from ultralytics import YOLO
import os
from PIL import Image


model=YOLO("best.pt")

print(model.names)

im1 = Image.open("test_coin.png")
im2 = Image.open("test_coin2.jpg")
im3 = Image.open("test_coin3.jpg")

results1 = model.predict(source=im1,conf=0.7, iou=0.45, save=True)

results2 = model.predict(source=im2,conf=0.7, iou=0.45, save=True)

results3 = model.predict(source=im3,conf=0.7, iou=0.45, save=True)