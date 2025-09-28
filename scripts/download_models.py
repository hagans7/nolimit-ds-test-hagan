# download_models.py
# scripts\download_models.py
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# Atur direktori cache di dalam image Docker
# Ini harus cocok dengan ENV VAR di Dockerfile
cache_dir = os.getenv("HF_HOME", "/app/huggingface_cache")

# Daftar model yang ingin diunduh
models_to_download = {
    "transformer_models": [
        "niejanee/tokopedia-sentiment-analysis-indobert"
    ],
    "sentence_transformer_models": [
        "firqaaa/indo-sentence-bert-base"
    ]
}

def download_all_models():
    """
    Mengunduh semua model yang terdaftar dan menyimpannya ke cache_dir.
    """
    print("--- Starting Model Download Process ---")

    # Unduh model transformers biasa
    for model_name in models_to_download["transformer_models"]:
        print(f"Downloading {model_name}...")
        try:
            AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
            print(f"✅ Successfully downloaded {model_name}")
        except Exception as e:
            print(f"❌ Failed to download {model_name}. Error: {e}")
    
    # Unduh model sentence-transformers
    for model_name in models_to_download["sentence_transformer_models"]:
        print(f"Downloading {model_name}...")
        try:
            SentenceTransformer(model_name, cache_folder=cache_dir)
            print(f"✅ Successfully downloaded {model_name}")
        except Exception as e:
            print(f"❌ Failed to download {model_name}. Error: {e}")

    print("--- Model Download Process Finished ---")

if __name__ == "__main__":
    download_all_models()