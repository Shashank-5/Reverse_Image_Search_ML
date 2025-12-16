import os

def get_next_search_folder(base_path="data/source"):
    os.makedirs(base_path, exist_ok=True)
    existing = sorted([d for d in os.listdir(base_path) if d.startswith("search_")])
    count = len(existing) + 1
    return os.path.join(base_path, f"search_{count}")
