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

def load_and_normalize(folder):
    # Load images
    envi_files, masked_files = load_images(folder)

    # Initialize lists to store normalized and masked data
    normalized_data_list = []
    masked_data_list = []

    for (hdr_file, data_file), mask_file in zip(envi_files, masked_files):
        # Load hyperspectral data
        img = open_image(os.path.join(folder, hdr_file))
        data = img.load().astype(np.float32)

        # Normalize the data
        normalized_data = normalize_data(data)

        # Load the mask
        mask = np.array(Image.open(os.path.join(folder, mask_file))) / 255.0

        # Apply the mask
        masked_data = apply_mask(normalized_data, mask)

        # Append to lists
        normalized_data_list.append(normalized_data)
        masked_data_list.append(masked_data)

    return normalized_data_list, masked_data_list
