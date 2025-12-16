import cv2
from ultralytics import YOLO
from train_test import model
from PIL import Image

im1 = Image.open("test_cat.jpeg")
results = model.predict(source=im1, save=True)