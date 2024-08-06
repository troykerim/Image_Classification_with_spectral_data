'''
This file is to perform any preprocessing on the HSI data before the PCA is implemented.
Such as normalization

'''
import os
import numpy as np
import warnings
import spectral
from PIL import Image
from spectral import open_image
from load_data import load_images

# spectral.settings.envi_support_nonlowercase_params = True
#warnings.filterwarnings('ignore', message='Unable to parse bad band list (bbl) in ENVI header as integers')

def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def apply_mask(hyperspectral_data, mask):
    masked_data = hyperspectral_data * mask[:, :, np.newaxis]
    return masked_data

def load_and_normalize(hdr_files, data_files, mask_files):
    # Initialize lists to store normalized and masked data
    normalized_data_list = []
    masked_data_list = []

    # Process hyperspectral data
    for hdr_file, data_file in zip(hdr_files, data_files):
        img = open_image(hdr_file)
        data = img.open_memmap()  # Using memory mapping to handle large files

        # Normalize the data
        normalized_data = normalize_data(data)
        normalized_data_list.append(normalized_data)

    # Process masks
    for mask_file in mask_files:
        mask = np.array(Image.open(mask_file)) / 255.0
        masked_data_list.append(mask)

    # Apply masks to normalized data
    final_masked_data_list = []
    for normalized_data, mask in zip(normalized_data_list, masked_data_list):
        masked_data = apply_mask(normalized_data, mask)
        final_masked_data_list.append(masked_data)

    return normalized_data_list, final_masked_data_list