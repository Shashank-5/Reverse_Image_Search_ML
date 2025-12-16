from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import os
import shutil
import pickle
import faiss

from app.model import extract_features_pil
from app.utils import get_next_search_folder

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Load FAISS index and metadata
index = faiss.read_index("faiss/resnet_faiss.index")
with open("faiss/resnet_meta.pkl", "rb") as f:
    meta = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search/", response_class=HTMLResponse)
async def search_image(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    query_feat = extract_features_pil(img)

    distances, indices = index.search(query_feat.reshape(1, -1), 10)
    results = [meta[i] for i in indices[0]]

    search_folder = get_next_search_folder()
    os.makedirs(search_folder, exist_ok=True)
    query_path = os.path.join(search_folder, "query.jpg")
    img.save(query_path)

    saved_paths = []
    for i, path in enumerate(results):
        dst = os.path.join(search_folder, f"match_{i+1}.jpg")
        shutil.copy(path, dst)
        saved_paths.append(dst)

    return templates.TemplateResponse("results.html", {
        "request": request,
        "query_path": "/" + query_path,
        "matches": ["/" + p for p in saved_paths]
    })

app.mount("/data", StaticFiles(directory="data"), name="data")
