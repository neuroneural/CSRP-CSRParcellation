import torch.nn as nn
from model.segunet import SegUnet
from monai.networks.nets import UNet
from monai.networks.nets import SwinUNETR
import torch
import torch.nn.functional as F
    
# Assume SegUnet is defined somewhere above

class SegmenterFactory:
    @staticmethod
    def get_segmenter(model_name, device):#, *args, **kwargs):
        """
        Factory method to instantiate segmentation models.

        Parameters:
        - model_name (str): Name of the model class to instantiate.
        - *args: Positional arguments to pass to the model class constructor.
        - **kwargs: Keyword arguments to pass to the model class constructor.

        Returns:
        - An instance of the requested segmentation model.
        """
        if model_name=="SegUnet":
            return SegUnet(
                c_in=1,
                c_out=3
            ).to(device)
        elif model_name=="MonaiUnet":
            return UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=3,
                channels=(64, 128, 256, 512, 1024),  # Larger capacity
                strides=(2, 2, 2, 2),
                num_res_units=3,  # Increase the number of residual units
                dropout=0.2,  # Optional: add dropout for regularization
            ).to(device)

        elif model_name=="SwinUNETR":
            return SwinUNETR(
                img_size=(192, 224, 192),  # Size of the input image
                in_channels=1,             # Number of input channels. MRI images are typically grayscale, so this is 1.
                out_channels=3,            # Number of output channels, corresponding to the segmentation classes.
                feature_size=48,           # Starting feature size. Consider adjusting based on computational resources and task complexity.
                use_checkpoint=True,       # Use gradient checkpointing to reduce memory consumption
                spatial_dims=3             # Number of spatial dimensions of the input images
            ).to(device)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    @staticmethod
    def pad_to_size(volume, target_size):
        _, _, d, h, w = volume.shape
        padding = [0, 0, 0, 0, 0, 0]  # Initialize padding [left, right, top, bottom, front, back]
        
        # Calculate padding needed to reach the target size
        padding[1] = target_size[4] - w  # right
        padding[3] = target_size[3] - h  # bottom
        padding[5] = target_size[2] - d  # back
        
        padded_volume = F.pad(volume, padding, "constant", 0)
        return padded_volume
    
    @staticmethod
    def crop_to_original_size(padded_volume, original_size):
        _, _, d, h, w = original_size
        cropped_volume = padded_volume[:, :, :d, :h, :w]
        return cropped_volume

    


# Example usage:
# Assuming 'device' is defined (e.g., 
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#segnet = SegmenterFactory.get_segmenter("SegUnet",device)
