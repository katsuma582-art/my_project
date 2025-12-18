import os
import cv2
import random
import albumentations as A


folder_path = "datasets/all_data/sr_images"
total_files = sum([len(files) for _, _, files in os.walk(folder_path)])
print(f"合計ファイル数: {total_files}")

for i in range(total_files):
    img_path = f'datasets/all_data/sr_images/sample{str(i).zfill(3)}.jpg'
    label_path = f'datasets/all_data/sr_labels/sample{str(i).zfill(3)}.txt'

    image = cv2.imread(img_path)
    if image is None:
        print(f"[WARN] 画像を読み込めません: {img_path} をスキップします")
        

    bboxes = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    # Assuming YOLO format: class x y w h
                    try:
                        bboxes.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"[WARN] ラベルファイルが見つかりません: {label_path} (空の bboxes として処理します)")

    #Downscale:downscale the image with zoom out and zoom in
    bboxes_for_aug = [bbox[1:] for bbox in bboxes]  # Extract x, y, w, h for augmentation
    result0 =A.Downscale(scale_range=(0.1, 0.25), p=1)(image=image,bboxes=bboxes_for_aug,BboxesParams=A.BboxParams(format='yolo'))

    out_img = result0['image']
    out_bboxes = result0.get('bboxes', [])

    out_img_path = f'datasets/all_data/augment_images/sample{str(total_files+1+i).zfill(3)}.jpg'
    cv2.imwrite(out_img_path, out_img)

    out_label_path = f'datasets/all_data/augment_labels/sample{str(total_files+1+i).zfill(3)}.txt'
    with open(out_label_path, 'w') as f:
        for i, bbox in enumerate(out_bboxes):
            class_id = bboxes[i][0] if i < len(bboxes) else 0  # Preserve original class
            # Clip bbox values to [0, 1]
            bbox = [max(0, min(1, b)) for b in bbox]
            f.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(class_id, *bbox))
            
            
    result1 = A.CoarseDropout(num_holes_range=(1, 100), hole_height_range=(8, 16), hole_width_range=(8, 16), p=1)(image=image,bboxes=bboxes_for_aug,BboxesParams=A.BboxParams(format='yolo'))    

    out_img = result1['image']
    out_bboxes = result1.get('bboxes', [])

    out_img_path = f'datasets/all_data/augment_images/sample{str(total_files+2+i).zfill(3)}.jpg'
    cv2.imwrite(out_img_path, out_img)

    out_label_path = f'datasets/all_data/augment_labels/sample{str(total_files+2+i).zfill(3)}.txt'
    with open(out_label_path, 'w') as f:
        for i, bbox in enumerate(out_bboxes):
            class_id = bboxes[i][0] if i < len(bboxes) else 0  # Preserve original class
            bbox = [max(0, min(1, b)) for b in bbox]
            f.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(class_id, *bbox))
            
    #Blur:apply blur to the image
    result2 = A.Blur()(image=image,bboxes=bboxes_for_aug,BboxesParams=A.BboxParams(format='yolo'))

    out_img = result2['image']
    out_bboxes = result2.get('bboxes', [])

    out_img_path = f'datasets/all_data/augment_images/sample{str(total_files+3+i).zfill(3)}.jpg'
    cv2.imwrite(out_img_path, out_img)

    out_label_path = f'datasets/all_data/augment_labels/sample{str(total_files+3+i).zfill(3)}.txt'
    with open(out_label_path, 'w') as f:
        for i, bbox in enumerate(out_bboxes):
            class_id = bboxes[i][0] if i < len(bboxes) else 0  # Preserve original class
            bbox = [max(0, min(1, b)) for b in bbox]
            f.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(class_id, *bbox))

    result3 = A.ChannelShuffle(p=1)(image=image,bboxes=bboxes_for_aug,BboxesParams=A.BboxParams(format='yolo'))

    out_img = result3['image']
    out_bboxes = result3.get('bboxes', [])

    out_img_path = f'datasets/all_data/augment_images/sample{str(total_files+4+i).zfill(3)}.jpg'
    cv2.imwrite(out_img_path, out_img)

    out_label_path = f'datasets/all_data/augment_labels/sample{str(total_files+4+i).zfill(3)}.txt'
    with open(out_label_path, 'w') as f:
        for i, bbox in enumerate(out_bboxes):
            class_id = bboxes[i][0] if i < len(bboxes) else 0  # Preserve original class
            bbox = [max(0, min(1, b)) for b in bbox]
            f.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(class_id, *bbox))

    result4 = A.ChannelDropout(channel_drop_range=(1, 1), p=1)(image=image,bboxes=bboxes_for_aug,BboxesParams=A.BboxParams(format='yolo'))

    out_img = result4['image']
    out_bboxes = result4.get('bboxes', [])

    out_img_path = f'datasets/all_data/augment_images/sample{str(total_files+5+i).zfill(3)}.jpg'
    cv2.imwrite(out_img_path, out_img)

    out_label_path = f'datasets/all_data/augment_labels/sample{str(total_files+5+i).zfill(3)}.txt'
    with open(out_label_path, 'w') as f:
        for i, bbox in enumerate(out_bboxes):
            class_id = bboxes[i][0] if i < len(bboxes) else 0  # Preserve original class
            bbox = [max(0, min(1, b)) for b in bbox]
            f.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(class_id, *bbox))

    result5 = A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1)(image=image,bboxes=bboxes_for_aug,BboxesParams=A.BboxParams(format='yolo'))

    out_img = result5['image']
    out_bboxes = result5.get('bboxes', [])

    out_img_path = f'datasets/all_data/augment_images/sample{str(total_files+6+i).zfill(3)}.jpg'
    cv2.imwrite(out_img_path, out_img)

    out_label_path = f'datasets/all_data/augment_labels/sample{str(total_files+6+i).zfill(3)}.txt'
    with open(out_label_path, 'w') as f:
        for i, bbox in enumerate(out_bboxes):
            class_id = bboxes[i][0] if i < len(bboxes) else 0  # Preserve original class
            bbox = [max(0, min(1, b)) for b in bbox]
            f.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(class_id, *bbox))

    transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    ], bbox_params=A.BboxParams(format='yolo'))

    for j in range(5):
        result6 = transform(image=image, bboxes=bboxes_for_aug)
        out_img = result6['image']
        out_bboxes = result6.get('bboxes', [])

        out_img_path = f'datasets/all_data/augment_images/sample{str(total_files+7+j+i).zfill(3)}.jpg'
        cv2.imwrite(out_img_path, out_img)

        out_label_path = f'datasets/all_data/augment_labels/sample{str(total_files+7+j+i).zfill(3)}.txt'
        with open(out_label_path, 'w') as f:
            for i, bbox in enumerate(out_bboxes):
                class_id = bboxes[i][0] if i < len(bboxes) else 0  # Preserve original class
                bbox = [max(0, min(1, b)) for b in bbox]
                f.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(class_id, *bbox))