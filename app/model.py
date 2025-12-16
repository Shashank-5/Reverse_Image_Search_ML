from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features_pil(img: Image.Image) -> np.ndarray:
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x).flatten().astype("float32")
    return features
