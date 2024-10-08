'''
Load images from a folder and store them into a list for later use in the program
'''
import os
import numpy as np
from PIL import Image
from spectral import open_image


# Old version of function
"""
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Add other formats if needed
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            images.append(img)
            # print(f"Loaded {len(images)} images from the folder: {folder}")
    return images"""


def load_images(folder):
    # Lists to store the 2 types images.
    #envi_files = [] # comment out if wanting to use specific file types
    masked_files = []
    
    # Uncomment if wanting to seperate the hdr files and data files into their own
    hdr_envi_files = []
    data_envi_files = []
    
    
    # Create a list of all files in the folder
    files = os.listdir(folder)
    
    # Sort files numerically
    files = sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x)) or -1))
    
    # For testing purposes comment this out, this will display all files in the input folder.
    # print(files)
    
    # Split our .hdr files, data files, and .png files
    hdr_files = [f for f in files if f.endswith('.hdr')]
    data_files = [f for f in files if not f.endswith('.hdr') and not f.endswith('.png')]
    png_files = [f for f in files if f.endswith('.png')]
    
    for hdr_file in hdr_files:
        base_name = hdr_file[:-4]
        data_file = next((f for f in data_files if f.startswith(base_name)), None)
        if data_file:
            # Comment this out if using the bottom 2.
            #envi_files.append((hdr_file, data_file))
            
            # If using 3 lists
            hdr_envi_files.append(hdr_file)
            data_envi_files.append(data_file)

    for png_file in png_files:
        masked_files.append(png_file)
        
    # Returns 2 lists. envi_files list contains 2 file types (.hdr and data-file)
    #return envi_files, masked_files

    # For 3 lists 
    return hdr_envi_files, data_envi_files, masked_files
    




