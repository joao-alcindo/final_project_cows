import os
import numpy as np
from scipy.ndimage import rotate
import torch
from torchvision import transforms

# Define the class for resizing with padding
class ResizeNpyWithPadding:
    def __init__(self, output_size):
        self.output_size = output_size
    
    def __call__(self, data):
        h, w = data.shape
        
        new_h, new_w = self.output_size
        top = (new_h - h) // 2
        bottom = new_h - h - top
        left = (new_w - w) // 2
        right = new_w - w - left
        
        resized_data = np.pad(data, ((top, bottom), (left, right)), mode='constant')
        return resized_data

# Define the class for random horizontal flipping
class RandomHorizontalFlipNpy:
    def __call__(self, data):
        if np.random.rand() < 0.5:
            data = np.fliplr(data)
        return data

# Define the class for random rotation
class RandomRotationNpy:
    def __init__(self, degrees):
        self.degrees = degrees
        
    def __call__(self, data):
        angle = np.random.uniform(self.degrees[0], self.degrees[1])
        rotated_data = rotate(data, angle, reshape=False, mode='constant', cval=0.0)
        return rotated_data