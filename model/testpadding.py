import torch
import torch.nn.functional as F

def pad_to_size(volume, target_size):
    _, _, d, h, w = volume.shape
    padding = [0, 0, 0, 0, 0, 0]  # Initialize padding [left, right, top, bottom, front, back]
    
    # Calculate padding needed to reach the target size
    padding[1] = target_size[4] - w  # right
    padding[3] = target_size[3] - h  # bottom
    padding[5] = target_size[2] - d  # back
    
    padded_volume = F.pad(volume, padding, "constant", 0)
    return padded_volume

def crop_to_original_size(padded_volume, original_size):
    _, _, d, h, w = original_size
    cropped_volume = padded_volume[:, :, :d, :h, :w]
    return cropped_volume

# Original and target sizes
original_size = [1, 1, 176, 208, 176]
target_size = [1, 1, 192, 224, 192]

# Assuming 'original_volume' is your original volume tensor
original_volume = torch.randn(original_size)  # Example tensor, replace with your actual volume

# Step 1: Pad the volume
padded_volume = pad_to_size(original_volume, target_size)

# Step 2: Crop back to the original size
cropped_volume = crop_to_original_size(padded_volume, original_size)

# Step 3: Assert no voxels have changed
assert torch.equal(original_volume, cropped_volume), "The voxels have been altered."

print("Assertion passed. No voxels were changed in the process.")

# Original and target sizes
original_size = [1, 3, 176, 208, 176]
target_size = [1, 3, 192, 224, 192]

# Assuming 'original_volume' is your original volume tensor
original_volume = torch.randn(original_size)  # Example tensor, replace with your actual volume

# Step 1: Pad the volume
padded_volume = pad_to_size(original_volume, target_size)

# Step 2: Crop back to the original size
cropped_volume = crop_to_original_size(padded_volume, original_size)

assert list(cropped_volume.size()) == original_size
# Step 3: Assert no voxels have changed
assert torch.equal(original_volume, cropped_volume), "The voxels have been altered."

print("Assertion passed. No voxels were changed in the process.")