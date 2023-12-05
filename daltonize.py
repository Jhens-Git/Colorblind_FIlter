import numpy as np
import cv2

# Process described here: https://ixora.io/projects/colorblindness/color-blindness-simulation-research/

def daltonize(image, blindtype):
    
    # Pick which transformation kernel to use
    if blindtype == "deuteranopia": # Kernel for deuteranopia
        kernel = np.array([[0.9, 0.4, 0.0],
                            [0.0, 0.8, 0.5],
                            [-0.3, 0.2, 1]])
               
    elif blindtype == "protanopia": # Kernel for protanopia
        kernel = np.array([[0.95, 0.35, 0.0],
                            [0, 0.9, 0.4],
                            [-0.2, 0.1, 1]])

    else: # Kernel for tritanopia
        kernel = np.array([[1, 0.183, 0],
                           [0, 0.927, 0.073],
                           [0, 0.5, 1]])

    # Remove gamma correction to processing colors ranging to [0, 1]
    gamma_removed = image / 255.0
    
    # Convert image colors using kernel colorspace 
    transformed = np.dot(gamma_removed, kernel.T)
    
    # Set the transformed image range to [0, 1]
    transformed = np.clip(transformed, 0, 1)
    
    # Implement gamma correction into the image
    gamma_corrected = transformed * 255
    
    # For handling error, set color range to [0, 225] range
    daltonized_image = np.clip(gamma_corrected, 0, 255).astype(np.uint8)
    
    return daltonized_image
