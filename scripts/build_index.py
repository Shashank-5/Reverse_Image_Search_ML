import faiss
import numpy as np

def build_index(features_path='features.npy', index_path='faiss/resnet_faiss.index'):
    features = np.load(features_path).astype('float32')
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    faiss.write_index(index, index_path)
    print("FAISS index saved to", index_path)

if __name__ == "__main__":
    build_index()
