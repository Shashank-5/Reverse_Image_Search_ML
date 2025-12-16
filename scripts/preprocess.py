import cv2
import os
from tqdm import tqdm

def preprocess_images(input_dir, output_dir, size=(300, 300)):
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_path = os.path.join(output_subdir, file)

                img = cv2.imread(input_path)
                if img is not None:
                    img_resized = cv2.resize(img, size)
                    cv2.imwrite(output_path, img_resized)

if __name__ == "__main__":
    input_directory = "data/raw/evaluation"
    output_directory = "data/preprocessed"
    preprocess_images(input_directory, output_directory)
