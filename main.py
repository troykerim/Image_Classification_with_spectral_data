import warnings
warnings.filterwarnings("ignore")
import os
from load_data import load_images
from preprocessing import load_and_normalize

if __name__ == "__main__":
    
    
    # Load in the data folder: 
    folder_path = '/home/troy/sweet_potato_data_combined'  
    
    hdr_envi_files, data_envi_files, masked_files = load_images(folder_path)
    
    # Testing the lists after load_data.py
    # Display the lengths of both lists
    #print(f"Number of ENVI files: {len(envi_files)}")
    print(f"Number of ENVI files: {len(hdr_envi_files)}")
    print(f"Number of data files: {len(data_envi_files)}")
    print(f"Number of masked files: {len(masked_files)}\n")
    print(f"Here an example of data file: {hdr_envi_files[500]}")
    print(f"Here an example of data file: {data_envi_files[1]}")
    print(f"Number of masked files: {masked_files[490]}\n")
    
    
    # Split envi_files into hdr_files and data_files
    hdr_files = [os.path.join(folder_path, hdr) for hdr, _ in hdr_envi_files]
    data_files = [os.path.join(folder_path, data) for _, data in data_envi_files]
    masked_files = [os.path.join(folder_path, mask) for mask in masked_files]
    
    # Pass the envi files and image mask to the preprocessing file
    normalized_data_list, masked_data_list = load_and_normalize(hdr_files, data_files, masked_files)
    
    
    # For displaying/testing purposes
    """
    images = load_images(folder_path)
    images = load_images(folder_path)
    print(f"Loaded {len(images)} images from the folder: '{folder_path}'")
    """

    
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
        print(f"PNG File: {png_file}")
    """