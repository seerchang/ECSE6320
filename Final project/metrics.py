import math

import numpy as np
from scipy.ndimage import sobel
from PIL import Image


def mse_metric(original, compressed):
    # original, compressed are numpy arrays of the same shape, RGB images
    # MSE = mean of (original - compressed)^2
    diff = (original.astype(np.float32) - compressed.astype(np.float32)) ** 2
    return diff.mean()


def psnr_metric(mse_val, max_val=255.0):
    if mse_val == 0:
        return float('inf')
    return 10 * math.log10((max_val**2) / mse_val)


def gradient_difference_metric(original_img, compressed_img):

    # Convert both images to grayscale (L)
    orig_gray = original_img.convert("L")
    comp_gray = compressed_img.convert("L")

    # Convert to numpy arrays
    orig_arr = np.asarray(orig_gray, dtype=float)
    comp_arr = np.asarray(comp_gray, dtype=float)

    # Compute gradients using Sobel filters
    orig_grad_x = sobel(orig_arr, axis=1)
    orig_grad_y = sobel(orig_arr, axis=0)
    comp_grad_x = sobel(comp_arr, axis=1)
    comp_grad_y = sobel(comp_arr, axis=0)

    # Gradient magnitudes
    orig_grad_mag = np.sqrt(orig_grad_x**2 + orig_grad_y**2)
    comp_grad_mag = np.sqrt(comp_grad_x**2 + comp_grad_y**2)

    # Compute the mean squared difference in gradient magnitude
    diff = (orig_grad_mag - comp_grad_mag)**2
    gdm = np.mean(diff)
    return gdm


