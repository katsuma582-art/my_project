from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import random

src_dir = 'datasets/coins/all_data/augment_images'
labels_src_dir = 'datasets/coins/all_data/augment_labels'

# collect available image files (keep extension) and sort
exts = ('.jpg', '.jpeg', '.png')
images = [f for f in sorted(os.listdir(src_dir)) if f.lower().endswith(exts)]
total_files = len(images)

# determine number of validation samples
k = int(total_files / 5)
if total_files > 0 and k <= 0:
	k = 1

if k > 0 and total_files > 0:
	valu_idx = list(random.sample(range(total_files), k))
else:
	valu_idx = []

train_idx = [i for i in range(total_files) if i not in valu_idx]

print('validation indices:', valu_idx)
print('train indices:', train_idx)

import shutil

# destination directories for validation (images and labels)
val_img_dst = 'datasets/coins/images/val'
val_lbl_dst = 'datasets/coins/labels/val'
train_img_dst = 'datasets/coins/images/train'
train_lbl_dst = 'datasets/coins/labels/train'

os.makedirs(val_img_dst, exist_ok=True)
os.makedirs(val_lbl_dst, exist_ok=True)
os.makedirs(train_img_dst, exist_ok=True)
os.makedirs(train_lbl_dst, exist_ok=True)

def move_sample_by_index(idx: int,img_dst: str,lbl_dst: str):
	"""Move image and corresponding label for images[idx] to val folders."""
	if idx < 0 or idx >= total_files:
		print(f"[WARN] index out of range: {idx}")
		return
	fname = images[idx]
	src_img_path = os.path.join(src_dir, fname)
	# label base name: drop extension and add .txt
	base, _ = os.path.splitext(fname)
	src_label_path = os.path.join(labels_src_dir, base + '.txt')

	if os.path.exists(src_img_path):
		shutil.move(src_img_path, os.path.join(img_dst, fname))
		print(f"moved image: {src_img_path} -> {img_dst}")
	else:
		print(f"[WARN] 画像が見つかりません: {src_img_path}")

	if os.path.exists(src_label_path):
		shutil.move(src_label_path, os.path.join(lbl_dst, base + '.txt'))
		print(f"moved label: {src_label_path} -> {lbl_dst}")
	else:
		print(f"[WARN] ラベルが見つかりません: {src_label_path}")

# move only selected validation samples
for idx in valu_idx:
	move_sample_by_index(idx, val_img_dst, val_lbl_dst)
 
for idx in train_idx:
    move_sample_by_index(idx, train_img_dst, train_lbl_dst)
    
