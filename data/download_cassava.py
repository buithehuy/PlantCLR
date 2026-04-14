import os
import subprocess

def download_cassava():
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Cassava")
    
    if os.path.exists(target_dir):
        print(f"Cassava dataset already exists at {target_dir}")
        return

    print("Downloading Cassava dataset using Kaggle API...")
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        subprocess.run(["kaggle", "competitions", "download", "-c", "cassava-leaf-disease-classification", "-p", target_dir], check=True)
        zip_path = os.path.join(target_dir, "cassava-leaf-disease-classification.zip")
        print("Extracting...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        os.remove(zip_path)
        print("Extraction complete.")
    except Exception as e:
        print(f"Download failed (Ensure kaggle API is set up properly): {e}")
        print("Please download 'cassava-leaf-disease-classification' from Kaggle and extract to data/Cassava/")

if __name__ == "__main__":
    download_cassava()
