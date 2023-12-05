import numpy as np
import cv2

def simulate(image, blindtype):
    
    # Pick which transformation kernel to use based on user's inputted blindtype
    if blindtype == "deuteranopia": # Kernel for deuteranopia
        kernel = np.array([[0.55, 0.45, 0],
                           [0, 0.45, 0.45],
                           [0, 0.525, 0.475]])
 
    elif blindtype == "protanopia": # Kernel for protanopia
        kernel = np.array([[0.95, 0.05, 0],
                           [0, 0.65, 0.35],
                           [0, 0.675, 0.325]])
        
    else: # Kernel for tritanopia
        kernel = np.array([[0.567, 0.433, 0],
                           [0.558, 0.442, 0],
                           [0, 0.242, 0.758]])

    # Transform the images
    simulated_image = cv2.transform(image, kernel)

    # Clip values to be in the valid range [0, 255]
    simulated_image = np.clip(simulated_image, 0, 255)

    # Convert back to uint8
    simulated_image = simulated_image.astype(np.uint8)

    return simulated_image
