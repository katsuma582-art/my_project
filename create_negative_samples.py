import os
import shutil
folder_path = "datasets/all_data/images"
total_files = sum([len(files) for _, _, files in os.walk(folder_path)])

# negative画像のパスをリストに追加
negative_images = [
    # 例: 'datasets/coins/negative_images/neg1.jpg',
    # ユーザーがここにパスを追加
] + [f'datasets\\all_data\\images\\sample{str(i).zfill(3)}.jpg' for i in range(total_files)]

for img_path in negative_images:
    if os.path.exists(img_path):
        # 画像をtrainにコピー
        # shutil.copy(img_path, 'datasets/coins/images/train/')
        # # ベース名を取得
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # # 空のラベルファイルを作成
        label_path = f'datasets/all_data/labels/{base_name}.txt'
        with open(label_path, 'w') as f:
            pass  # 空ファイル
        print(f'Copied {img_path} to train and created empty label {label_path}')
    else:
        print(f'Image not found: {img_path}')