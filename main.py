'''import os
from PIL import Image '''
import os
import sys
import spectral
from load_data import load_images

if __name__ == "__main__":
    folder_path = '/home/troy/sweet_potato_data_combined'  # Path to your folder
    
    #images = load_images(folder_path)
    #images = load_images(folder_path)
    # print(f"Loaded {len(images)} images from the folder: '{folder_path}'")
    
    envi_files, masked_files = load_images(folder_path)

    # Display the lengths of both lists
    print(f"Number of ENVI files: {len(envi_files)}")
    print(f"Number of masked files: {len(masked_files)}\n")
    """
    # Display the contents of both lists
    print("ENVI files:")
    print("------------------------")
    for hdr_file, data_file in envi_files:
        print(f"Header File: {hdr_file}")
        print(f"Data File: {data_file}")

    print("\nMasked files:")
    print("------------------------")
    for png_file in masked_files:
        print(f"PNG File: {png_file}")"""