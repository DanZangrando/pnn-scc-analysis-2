import os
import sys
import zipfile
import urllib.request
import subprocess

def download_file(url, dest_path, desc=""):
    st = None
    if "streamlit" in sys.modules:
        import streamlit as st
        
    if st:
        st.info(f"⬇️ Descargando {desc} de Zenodo... (esto puede tardar unos minutos)")
        
    print(f"Downloading {url} to {dest_path}...")
    urllib.request.urlretrieve(url, dest_path)

def patch_scipy_deprecation():
    target_file = "src/counting_perineuronal_nets/methods/density/target_builder.py"
    if os.path.exists(target_file):
        with open(target_file, "r") as f:
            content = f.read()
        if "from scipy.signal import gaussian" in content:
            print("Applying SciPy 1.14+ gaussian import patch...")
            content = content.replace("from scipy.signal import gaussian", "from scipy.signal.windows import gaussian")
            with open(target_file, "w") as f:
                f.write(content)

def ensure_models_and_code():
    # 1. Ensure src directory and clone if needed
    if not os.path.exists("src/counting_perineuronal_nets"):
        print("Cloning counting_perineuronal_nets repository...")
        st = None
        if "streamlit" in sys.modules:
            import streamlit as st
        if st:
            st.info("🛰️ Clonando repositorio oficial de counting_perineuronal_nets...")
            
        os.makedirs("src", exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/ciampluca/counting_perineuronal_nets", "src/counting_perineuronal_nets"], check=True)
        
        # Apply the scipy 1.14+ deprecation patch immediately
        patch_scipy_deprecation()
    else:
        # Just double check the patch is applied even if the folder exists
        patch_scipy_deprecation()

    # 2. Ensure models directory exists
    os.makedirs("data/models", exist_ok=True)

    # 3. Check and download localization model
    loc_model_pth = "data/models/pnn_v2_fasterrcnn_640/best.pth"
    if not os.path.exists(loc_model_pth):
        zip_path = "data/models/pnn_v2_fasterrcnn_640.zip"
        url = "https://zenodo.org/records/7985860/files/pnn_v2_fasterrcnn_640.zip?download=1"
        download_file(url, zip_path, "Modelo de Localización (Faster R-CNN)")
        
        st = None
        if "streamlit" in sys.modules:
            import streamlit as st
        if st:
            st.info("📦 Descomprimiendo modelo de localización...")
            
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/models/")
            
        if os.path.exists(zip_path):
            os.remove(zip_path)

    # 4. Check and download scoring model
    score_model_pth = "data/models/pnn_v2_scoring_rank_learning/best.pth"
    if not os.path.exists(score_model_pth):
        zip_path = "data/models/pnn_v2_scoring_rank_learning.zip"
        url = "https://zenodo.org/records/7985860/files/pnn_v2_scoring_rank_learning.zip?download=1"
        download_file(url, zip_path, "Modelo de Calificación (PNNscore)")
        
        st = None
        if "streamlit" in sys.modules:
            import streamlit as st
        if st:
            st.info("📦 Descomprimiendo modelo de calificación...")
            
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/models/")
            
        if os.path.exists(zip_path):
            os.remove(zip_path)
