"""
Downloads model weights from Google Drive if not present locally.
Called at app startup on Streamlit Cloud.
"""

import os
import gdown


WEIGHTS = {
    "weights/smoke_classifier_finetuned.pth": "1DnlztcFi6L61-ehavWZ0e_RlOb9rZjqT",
    "weights/G_hazy2clear_lite.pth":          "1iwjYbvVRozGLcNw8EylXd7sLPXZ3jy0M",
}


def ensure_weights():
    """Download any missing weight files from Drive."""
    os.makedirs("weights", exist_ok=True)
    for local_path, file_id in WEIGHTS.items():
        if not os.path.exists(local_path):
            print(f"Downloading {local_path}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, local_path, quiet=False)
            print(f"  Done.")
        else:
            print(f"Found: {local_path}")