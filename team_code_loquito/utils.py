import numpy as np
import torch
from torchvision import transforms
from collections import deque

def convert_numpy_image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Converts a NumPy image array to a normalized PyTorch tensor.

    This function takes an image in NumPy format with shape (H, W, C),
    in BGR channel order and uint8 type, and converts it to a PyTorch tensor 
    with shape (C, H, W), in RGB order, float32 type, and pixel values 
    normalized to the [0, 1] range.

    Args:
        image (np.ndarray): Input image as a NumPy array of shape (H, W, C),
            with BGR channel order and dtype uint8.

    Returns:
        torch.Tensor: Converted image as a PyTorch tensor of shape (C, H, W),
            with RGB channel order, dtype float32, and values in the [0, 1] range.
    """
    torch_image = torch.from_numpy(image)[:,:,[2,1,0]].permute(2, 0, 1).float() / 255.0
    return torch_image


def get_images_tensor(images_tensor: torch.Tensor, img_size=(224, 224), normalize=True, augment=False, random_seed=None) -> torch.Tensor:
    """
    Processes a batch of image tensors with optional resizing, normalization, and augmentation.

    This function takes a 4D image tensor of shape [B, C, H, W] and applies resizing, 
    optional data augmentation (color jitter), and normalization.

    Args:
        images_tensor (torch.Tensor): A batch of images as a 4D tensor with shape [B, C, H, W].
        img_size (tuple, optional): Target size (height, width) for resizing. Defaults to (224, 224).
        normalize (bool, optional): Whether to normalize the images using ImageNet statistics.
            Defaults to True.
        augment (bool, optional): Whether to apply random color augmentations (brightness, contrast, 
            saturation, hue). Defaults to False.
        random_seed (int, optional): Seed for reproducibility of augmentations. Defaults to None.

    Returns:
        torch.Tensor: Transformed batch of images with shape [B, C, H, W].
    """
    # Add resizing to the transforms
    resize_transform = transforms.Resize(img_size, antialias=True)
    images_tensor = resize_transform(images_tensor)

    # Apply augmentations if required
    if augment:
        rng = torch.Generator()
        if random_seed is not None:
            rng.manual_seed(random_seed)
        
        # Define a set of color transformations
        brightness_factor, contrast_factor, saturation_factor, hue_factor = torch.rand(4, generator=rng)
        brightness_factor = brightness_factor + 0.5
        contrast_factor = contrast_factor + 0.5
        saturation_factor = saturation_factor + 0.5
        hue_factor = hue_factor*0.1 - 0.05
        
        images_tensor = transforms.functional.adjust_brightness(images_tensor, brightness_factor)
        images_tensor = transforms.functional.adjust_contrast(images_tensor, contrast_factor)
        images_tensor = transforms.functional.adjust_saturation(images_tensor, saturation_factor)
        images_tensor = transforms.functional.adjust_hue(images_tensor, hue_factor)

    # Normalize the images if required
    if normalize:
        normalization_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        images_tensor = normalization_transform(images_tensor)

    return images_tensor

# PID Controller from TransFuser
class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)

    def step(self, error):
        self._window.append(error)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative