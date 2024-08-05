'''import os
from PIL import Image '''
from load_data import load_images_from_folder

if __name__ == "__main__":
    folder_path = '/home/troy/sweet_potato_data_combined'  # Path to your folder
    images = load_images_from_folder(folder_path)
    
    print(f"Loaded {len(images)} images from the folder: '{folder_path}'")