import torch
import spectral
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import warnings
import logging
import torch.nn.functional as F
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
import time
from collections import Counter
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Suppress warnings from the spectral library by setting the logging level to ERROR
logging.getLogger('spectral').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Unable to parse bad band list.*")

# Handle non-lowercase parameter warning
spectral.settings.envi_support_nonlowercase_params = True

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#dirpath = '/home/jovyan/sweet_potato_dataset' # original dataset
dirpath = '/home/jovyan/dataset_tester' 
#dirpath = '/home/jovyan/data_aug_03_temp_part4'
print(dirpath)

def split_files(dirpath):
    """
    Splits and categorizes the contents of a directory into .hdr files, data files, and .png mask files.

    Parameters:
    ----------
    dirpath : str
        The full path to the directory containing the hyperspectral dataset.

    Returns:
    -------
    tuple of lists:
        - hdr_files : list of str
            Filenames ending with '.hdr', representing header files.
        - data_files : list of str
            Filenames that are not '.hdr' or '.png' files, assumed to be the associated hyperspectral data files (with no extension).
        - mask_files : list of str
            Filenames ending with '.png', representing binary mask images.

    Notes:
    -----
    - Files are sorted numerically based on digits in their names.
    - Non-file entries like subdirectories are not filtered out; this function assumes all entries in the directory are relevant files.
    """
    # Get a list of all files in the folder
    files = os.listdir(dirpath)
    
    # Sort files numerically begining at 1
    files = sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x)) or -1))
    
    # Filter out .hdr files, data files, and .png files into their own list
    hdr_files = [f for f in files if f.endswith('.hdr')]
    data_files = [f for f in files if not f.endswith('.hdr') and not f.endswith('.png')]
    mask_files = [f for f in files if f.endswith('.png')]
    
    # Return Each file type as a list
    return hdr_files, data_files, mask_files
hdr_files, data_files, mask_files = split_files(dirpath)

def group_files(hdr_files, data_files, mask_files):
    """
    Groups corresponding .hdr, data, and mask (.png) files into tuples for each spectral image.

    Parameters:
    ----------
    hdr_files : list of str
        List of filenames ending in '.hdr', representing header files.
    
    data_files : list of str
        List of filenames representing hyperspectral data files (no extension).
    
    mask_files : list of str
        List of filenames ending in '.png', representing binary mask images.

    Returns:
    -------
    paired_files : list of tuples
        Each tuple contains the full paths to the corresponding (.hdr, data, .png) files 
        for a single hyperspectral image, in the form: (hdr_path, data_path, mask_path).

    Notes:
    -----
    - Assumes that all three file types share a common base filename.
    - Only file sets for which all three components exist are included.
    - Requires the global variable `dirpath` to be defined to construct full file paths.
    """
    #hdr_files, data_files, mask_files = split_files(dirpath)
    paired_files = []
    for hdr_file in hdr_files:
        base_name = hdr_file[:-4]
        data_file = next((f for f in data_files if f.startswith(base_name)), None)
        mask_file = next((f for f in mask_files if f.startswith(base_name)), None)
        if data_file and mask_file:
            paired_files.append((os.path.join(dirpath, hdr_file),
                                 os.path.join(dirpath, data_file),
                                 os.path.join(dirpath, mask_file)))
    return paired_files
#Display each spectral image with its 3 files
paired_files = group_files(hdr_files, data_files, mask_files)


def load_and_resize_data(hdr_file, data_file, mask_file):
    # Load hyperspectral data
    img = spectral.open_image(hdr_file)
    data = img.load()

    # band indices start from 0 when view the .HDR file
    # Band 23 = 634 nm
    # Band 25 = 650 nm
    # Band 45 = 810 nm
    # Band 50 = 850 nm
    selected_bands = data[:, :, [23, 45]]  

    # Resize the selected bands
    resized_data = torch.nn.functional.interpolate(
        torch.tensor(selected_bands).permute(2, 0, 1).unsqueeze(0),  # Convert to tensor and prepare for resizing
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).permute(1, 2, 0)  # Convert back to original shape

    # Load and resize the mask
    mask = plt.imread(mask_file)  # Assuming mask is a .png image
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    resized_mask = torch.nn.functional.interpolate(
        mask_tensor.unsqueeze(0),
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)
    return resized_data, resized_mask
resized_data, resized_mask = load_and_resize_data(paired_files[0][0], paired_files[0][1], paired_files[0][2])



def find_roi_pixels(mask):
    # Ensure mask is a PyTorch tensor
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask)
    
    # Remove any singleton dimensions (e.g., [1, H, W] -> [H, W])
    mask = mask.squeeze()

    # Convert mask to binary (white pixels as 1, everything else as 0)
    binary_mask = (mask > 0.5).float()

    # Find coordinates of white pixels
    roi_coords = torch.nonzero(binary_mask, as_tuple=False)

    # Ensure roi_coords has exactly 2 columns (row, col) by checking the mask's shape
    if roi_coords.shape[1] > 2:
        roi_coords = roi_coords[:, :2]  # Take only the row and col indices

    return roi_coords.numpy()  # Convert to numpy if needed


def extract_roi_spectral_data(hyperspectral_data, roi_coords):
    roi_spectral_data = hyperspectral_data[roi_coords[:, 0], roi_coords[:, 1], :]
    return roi_spectral_data
# Find ROI coordinates
roi_coords = find_roi_pixels(resized_mask.numpy())
roi_spectral_data = extract_roi_spectral_data(resized_data, roi_coords)


def calculate_ndvi(roi_spectral_data):
    # Assume band 0 is Red and band 1 is NIR in the resized_data
    red_band = roi_spectral_data[:, 0]
    nir_band = roi_spectral_data[:, 1]

    # Calculate NDVI
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)  # Adding epsilon to avoid division by zero
    return ndvi


def get_label_name(label):
    labels = [
        "Healthy",
        "Moderately Healthy",
        "Early Necrosis",
        "Moderate Necrosis",
        "Severe Necrosis",
        "Dead or Inanimate Object"
    ]
    return labels[label]


def assign_label_from_ndvi(ndvi_value):
    if ndvi_value > 0.6:
        return 0  # Healthy
    elif 0.5 <= ndvi_value < 0.59:      
        return 1  # Moderately Healthy
    elif 0.4 <= ndvi_value < 0.49:
        return 2  # Early Necrosis
    elif 0.3 <= ndvi_value < 0.39:
        return 3  # Moderate Necrosis
    elif 0.05 <= ndvi_value < 0.29:  # Defualt 0.05
        return 4  # Severe Necrosis
    else:
        return 5  # Dead or Inanimate Object
		

class SweetPotatoDataset(Dataset):
    def __init__(self, data_list, augment_minority=False, augment_factor=3):
        self.data_list = data_list
        self.augment_minority = augment_minority
        self.augment_factor = augment_factor
        
        # augmentation transformations
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])
        
        # Augment the minority classes if specified
        if self.augment_minority:
            self.data_list = self._augment_minority_classes(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data, mask = self.data_list[idx]
        
        # Process hyperspectral data
        data = data.permute(2, 0, 1).unsqueeze(1)  # [2, 1, 224, 224] (depth=1)

        # Process mask
        mask = mask.squeeze(0).unsqueeze(0).unsqueeze(1)  # [1, 1, 224, 224] (depth=1)

        # Concatenate data and mask along the channel dimension
        combined_tensor = torch.cat((data, mask), dim=0)  # [3, 1, 224, 224] (channels=3, depth=1)

        # Calculate NDVI-based label
        roi_coords = find_roi_pixels(mask.squeeze().numpy())
        roi_spectral_data = extract_roi_spectral_data(data.squeeze().permute(1, 2, 0).numpy(), roi_coords)
        ndvi_values = calculate_ndvi(roi_spectral_data)

        # Assign label based on NDVI
        mean_ndvi = ndvi_values.mean().item()
        label = assign_label_from_ndvi(mean_ndvi)  # Assign one of the 6 labels

        if label is None:  # Handle cases where the label is invalid
            raise ValueError(f"Invalid NDVI value {mean_ndvi} for index {idx}")
        
        label = torch.tensor(label, dtype=torch.long)

        # Apply augmentations if augment_minority is True and the label is a minority class
        if self.augment_minority and label in [3, 5]:  
            combined_tensor = self.augmentations(combined_tensor)

        return combined_tensor, label

    def _augment_minority_classes(self, data_list):
        # Count the occurrences of each label to determine minority classes
        label_counts = Counter()
        augmented_data_list = []

        # Calculate labels for all data in the list to count class occurrences
        for data, mask in data_list:
            combined_tensor, label = self._get_combined_tensor_and_label(data, mask)
            label_counts[label.item()] += 1

        # Identify minority classes based on label counts
        minority_classes = [label for label, count in label_counts.items() if count < max(label_counts.values())]

        # Augment minority classes
        for data, mask in data_list:
            combined_tensor, label = self._get_combined_tensor_and_label(data, mask)
            if label.item() in minority_classes:
                # Append the original and multiple augmented versions
                augmented_data_list.append((data, mask))
                for _ in range(self.augment_factor):
                    augmented_data = self.augmentations(combined_tensor)
                    augmented_data_list.append((augmented_data[:2], augmented_data[-1].unsqueeze(0)))  # Separate augmented data and mask
            else:
                # Append the original sample for majority classes
                augmented_data_list.append((data, mask))

        return augmented_data_list

    def _get_combined_tensor_and_label(self, data, mask):
        # Process data and mask to calculate label (similar to __getitem__)
        data = data.permute(2, 0, 1).unsqueeze(1)
        mask = mask.squeeze(0).unsqueeze(0).unsqueeze(1)
        combined_tensor = torch.cat((data, mask), dim=0)
        roi_coords = find_roi_pixels(mask.squeeze().numpy())
        roi_spectral_data = extract_roi_spectral_data(data.squeeze().permute(1, 2, 0).numpy(), roi_coords)
        ndvi_values = calculate_ndvi(roi_spectral_data)
        mean_ndvi = ndvi_values.mean().item()
        label = assign_label_from_ndvi(mean_ndvi)
        return combined_tensor, torch.tensor(label, dtype=torch.long)
		
class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # Dropout layer with a probability of 0.5 is default
        self.dropout = nn.Dropout(p=0.4)

        # 7 Hidden Layers
        self.fc1 = nn.Linear(128 * 1 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, 256) # Hidden Layer 1
        self.fc3 = nn.Linear(256, 256) 
        self.fc4 = nn.Linear(256, 256) 
        self.fc5 = nn.Linear(256, 256) 
        self.fc6 = nn.Linear(256, 256) 
        self.fc7 = nn.Linear(256, 256) 
        # self.fc8 = nn.Linear(256, 256)
        self.fc9 = nn.Linear(256, 6)  # Output layer (6 classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        x = F.relu(self.fc7(x))
        x = self.dropout(x)
        #x = F.relu(self.fc8(x))
        #x = self.dropout(x)
        x = self.fc9(x)
        return x


resized_data_list = []
resized_mask_list = []

# Process each file pair to create resized data and mask tensors
for hdr_file, data_file, mask_file in paired_files:
    resized_data, resized_mask = load_and_resize_data(hdr_file, data_file, mask_file)
    resized_data_list.append(resized_data)
    resized_mask_list.append(resized_mask)
paired_tensors = []
for i in range(len(resized_data_list)):
    paired_tensors.append((resized_data_list[i], resized_mask_list[i]))
	
# Cross-validation loop with Early Stopping
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# Number of folds (K)
k = 5

# Initialize KFold
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Lists to store accuracies and confusion matrices for each fold
fold_accuracies = []
confusion_matrices = []

# Variables to track hyperparameters
epoch_counts = []          # Number of epochs completed for each fold (including early stopping)
initial_weights_list = []  # Initial weights for each fold
final_weights_list = []    # Final weights for each fold

fold = 1
for train_index, test_index in kf.split(paired_tensors):
    print(f"Fold {fold}/{k}")

    # Hyperparemeters
    batch_size = 16
    patience = 5 
    lr = 0.0001 
    weight_decay = 5e-4
    num_epochs = 80    # 50
    
    weights = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.05, 0.3]).to(device) #Also gives 84-85%
    #weights = torch.tensor([0.127310694898272, 0.11882331523838718, 0.21713905322750954, 0.27958427114914625, 0.13798836608328832, 0.1191542994033966]).to(device)
    # weights = torch.tensor([0.15, 0.2, 0.2, 0.25, 0.05, 0.15]).to(device) Gives 84-85%
    #weights = torch.tensor([0.15, 0.2, 0.2, 0.2, 0.05, 0.2]).to(device)  OLD

    # Split the data into training and testing sets for the current fold
    train_data = [paired_tensors[i] for i in train_index]
    test_data = [paired_tensors[i] for i in test_index]
    
    # Create datasets for the current fold
    train_dataset = SweetPotatoDataset(train_data)
    test_dataset = SweetPotatoDataset(test_data)

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, criterion, optimizer for this fold
    model = Simple3DCNN().to(device)

    # Modify Weights - Assign higher weights at indices 3 and 5
   # weights = torch.tensor([0.1, 0.15, 0.15, 0.25, 0.1, 0.25]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)

    # Early Stopping parameters
    best_loss = float('inf')
    epochs_no_improve = 0

    # Save initial weights for this fold
    initial_weights_list.append(weights.cpu().numpy().tolist())

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Track time for each epoch
        epoch_start_time = time.time()

        # Training loop
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        scheduler.step(avg_val_loss)
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f'Epoch [{epoch + 1}/{num_epochs}], Fold [{fold}/{k}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_duration:.2f}s')

        # Early Stopping Check
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()  # Save the best model
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f'Early stopping at epoch {epoch + 1} for Fold {fold}/{k}')
            model.load_state_dict(best_model_state)  # Load the best model state
            epoch_counts.append(epoch + 1)  # Track the number of epochs run before stopping
            break
    if epochs_no_improve < patience:
        epoch_counts.append(num_epochs)

    # Post-training evaluation for this fold
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        # Evaluate model on test data
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        # Save final weights - Selecting only the first 6 final weights from fc6 for display
        final_weights = model.fc6.weight.detach().cpu().numpy().flatten()[:6]
        final_weights_list.append(final_weights.tolist())

        # Calculate accuracy for the fold
        accuracy = 100 * correct / total
        fold_accuracies.append(accuracy)
        print(f'Fold {fold}/{k} Test Accuracy: {accuracy:.2f}%')

        # Confusion matrix for the fold
        cm = confusion_matrix(all_labels, all_predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        confusion_matrices.append(cm_normalized)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues", 
                    xticklabels=["Healthy", "Moderately Healthy", "Early Necrosis", "Moderate Necrosis", "Severe Necrosis", "Dead or Inanimate Object"], 
                    yticklabels=["Healthy", "Moderately Healthy", "Early Necrosis", "Moderate Necrosis", "Severe Necrosis", "Dead or Inanimate Object"])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - Fold {fold}')
        plt.show()

    # Move to the next fold
    fold += 1

# After cross-validation: Calculate and display average accuracy and consolidated confusion matrix
avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)

# Count the number of fully connected layers starting from fc2
num_fc_layers = len([layer for layer in dir(Simple3DCNN()) if layer.startswith('fc')]) - 1

# Display Model Results:
print(f"\n=== Model Summary ===")
print(f"Batch Size: {batch_size}")
print(f"Learning Rate: {lr}")

# Display initial class weights with 2 decimal formatting
initial_weights_to_display = [weights[:6] for weights in initial_weights_list]
print("\nInitial Class Weights for CrossEntropyLoss (showing up to 6 values per fold):")
for i, weights in enumerate(initial_weights_to_display):
    formatted_weights = ', '.join(f"{w:.2f}" for w in weights)
    print(f"Fold {i + 1}: [{formatted_weights}]")

# Display final class weights and accuracy for each fold
final_weights_to_display = [weights[:6] for weights in final_weights_list]
print("\nFinal Class Weights at End of Training and Accuracy per Fold (showing up to 6 values per fold):")
for i, (weights, accuracy) in enumerate(zip(final_weights_to_display, fold_accuracies)):
    formatted_weights = ', '.join(f"{w:.3f}" for w in weights)
    print(f"Fold {i + 1}: Weights: [{formatted_weights}], Accuracy: {accuracy:.2f}%")

print(f"Patience (Early Stopping): {patience}")
print(f"Weight Decay: {weight_decay}")
print(f"Total Number of Epochs: {num_epochs}")
print(f"Number of Epochs completed with Early Stopping per fold: {epoch_counts}")
print(f"Number of Fully Connected Layers (excluding fc1): {num_fc_layers}")
print(f"Dropout Probability: {model.dropout.p}")
print(f"Average Accuracy across {k} folds: {avg_accuracy:.2f}%")

# Consolidate confusion matrices into a single matrix
consolidated_cm = np.sum(confusion_matrices, axis=0)
consolidated_cm_norm = consolidated_cm.astype('float') / consolidated_cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(consolidated_cm_norm, annot=True, fmt=".2%", cmap="Blues", 
            xticklabels=["Healthy", "Moderately Healthy", "Early Necrosis", "Moderate Necrosis", "Severe Necrosis", "Dead or Inanimate Object"], 
            yticklabels=["Healthy", "Moderately Healthy", "Early Necrosis", "Moderate Necrosis", "Severe Necrosis", "Dead or Inanimate Object"])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Consolidated Confusion Matrix - All Folds')
plt.show()