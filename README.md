# Reverse Image Search – Food Image Retrieval System

This project is a **reverse image search engine** for food images, built using **FastAPI**, **ResNet50**, **FAISS**, and **Jinja2** templating. Given a food image, it finds and displays the top 10 visually similar images from a dataset.

# Create Virtual Environment

```bash
python3 -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
```

## Install Dependencies
pip install -r requirements.txt

## Run command 
python -m uvicorn app.main:app --reload


##  Features

-  Upload a food image via web interface
-  Extract ResNet50 features
-  Search using FAISS index
-  Display top 10 visually similar images
-  Saves results in `data/source/search_x/` folders

##  Scripts Usage

- Preprocess images:
  ```bash
  python scripts/preprocess.py
  ```

- Extract features:
  ```bash
  python scripts/extract_features.py
  ```

- Build FAISS index:
  ```bash
  python scripts/build_index.py
