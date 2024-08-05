'''
Load images from a folder and store them into a list for later use in the program
'''
import os
from PIL import Image

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Add other formats if needed
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            images.append(img)
            # print(f"Loaded {len(images)} images from the folder: {folder}")
    return images
