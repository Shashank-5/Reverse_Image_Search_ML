from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np, os, pickle
from tqdm import tqdm

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_resnet50_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

def extract_features(image_dir, output_path='features.npy', meta_path='faiss/resnet_meta.pkl'):
    all_features = []
    meta = {}
    idx = 0
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                print(f"📷 Checking: {img_path}")  # Add this line

                features = extract_resnet50_features(img_path)
                if features is not None:
                    print(f"✅ Extracted: {img_path}")
                    all_features.append(features)
                    meta[idx] = img_path
                    idx += 1
                else:
                    print(f"❌ Failed: {img_path}")


    np.save(output_path, np.array(all_features).astype('float32'))
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

if __name__ == "__main__":
    extract_features("data/preprocessed")
