import os
import urllib.request
import zipfile

def download_and_extract():
    url = "https://data.mendeley.com/datasets/tywbtsjrjv/1/files/b4e3a32f-c0bd-4060-81e9-6144231f2520/PlantVillage.zip"
    target_dir = os.path.dirname(os.path.abspath(__file__))
    pv_dir = os.path.join(target_dir, "PlantVillage")
    zip_path = os.path.join(target_dir, "PlantVillage.zip")
    
    if os.path.exists(pv_dir):
        print(f"PlantVillage dataset already exists at {pv_dir}")
        return
        
    print(f"Downloading PlantVillage from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print("Extraction complete.")
        os.remove(zip_path)
    except Exception as e:
        print(f"Failed to automatically download: {e}")
        print("Please download manually and place the 'train', 'val', 'test' folders inside data/PlantVillage/")

if __name__ == "__main__":
    download_and_extract()
