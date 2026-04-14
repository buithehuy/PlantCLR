import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

def reorganize():
    base_dir = "/home/huy/Research/SSL/PlantPathology/data/Cassava/cassava-leaf-disease-classification"
    target_base = "/home/huy/Research/SSL/PlantPathology/data/Cassava"
    train_csv = os.path.join(base_dir, "train.csv")
    
    if not os.path.exists(train_csv):
        print(f"Could not find {train_csv}")
        return
        
    df = pd.read_csv(train_csv)
    # 80/20 split stratify to maintain class balance
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    def move_files(df, split_name):
        split_dir = os.path.join(target_base, split_name)
        for _, row in df.iterrows():
            img_id = row['image_id']
            label = str(row['label'])
            class_dir = os.path.join(split_dir, label)
            os.makedirs(class_dir, exist_ok=True)
            
            src = os.path.join(base_dir, "train_images", img_id)
            dst = os.path.join(class_dir, img_id)
            if os.path.exists(src) and not os.path.exists(dst):
                # Using move is instant and saves space compared to copy
                shutil.move(src, dst)
                
    print("Moving files into train/ and val/ datasets...")
    move_files(train_df, 'train')
    move_files(val_df, 'val')
    print("Reorganization complete!")

if __name__ == "__main__":
    reorganize()
