import sys
import numpy as np
from numpy.fft import fft2, fftshift, ifftshift
from scipy.interpolate import interp1d
import scipy.io
from PIL import Image
import cv2 # OpenCV library for image processing

def scale_and_pad(img, target_shape, pad_value=0):
    """
    Scales an image to fit within target_shape while maintaining aspect ratio,
    and pads the remaining area.

    Args:
        img (np.ndarray): The input image (grayscale or color).
        target_shape (tuple): The desired output shape (height, width).
        pad_value (int or tuple): The value to use for padding (e.g., 0 for black).
                                   Should be a tuple (B, G, R) for color images if
                                   a specific color other than black is needed.

    Returns:
        np.ndarray: The scaled and padded image with shape target_shape.
    """
    target_h, target_w = target_shape
    original_h, original_w = img.shape[:2] # Works for both grayscale and color

    # Calculate scaling factors for height and width
    scale_h = target_h / original_h
    scale_w = target_w / original_w

    # Choose the smaller scaling factor to ensure the image fits entirely
    scale_factor = min(scale_h, scale_w)

    # Calculate new dimensions
    new_w = int(round(original_w * scale_factor))
    new_h = int(round(original_h * scale_factor))

    # Resize the image using the calculated scale factor
    # cv2.resize expects (width, height) for dsize
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_LINEAR)

    # Create the target canvas with the specified padding value
    # Check if the image is color or grayscale
    if img.ndim == 3 and resized_img.ndim == 3: # Color image
        target_canvas = np.full((target_h, target_w, img.shape[2]), pad_value, dtype=img.dtype)
        # Ensure resized_img also has 3 channels if original did (robustness)
        if resized_img.shape[2] != img.shape[2]:
             # This shouldn't happen with standard cv2.resize, but good practice
             print("Warning: Channel mismatch after resize.")
             # Attempt to fix or raise error depending on need
             if img.shape[2] == 3 and resized_img.ndim==2: # if resized became grayscale somehow
                 resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)


    elif img.ndim == 2 and resized_img.ndim == 2: # Grayscale image
        target_canvas = np.full((target_h, target_w), pad_value, dtype=img.dtype)
    else:
        raise ValueError(f"Inconsistent image dimensions. Original: {img.ndim}D, Resized: {resized_img.ndim}D")


    # Calculate padding amounts
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    # Place the resized image onto the canvas (centered)
    # Slicing indices: [start_row : end_row, start_col : end_col]
    target_canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized_img

    return target_canvas

# Utility: wrap an angle array to [-pi, pi]
def wrapToPi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Loads the SLM lookup table from the .mat file and computes phaseVal and SLMDigit.
def SLM_LUT():
    # Load the MAT file containing 'totalSum'
    mat_data = scipy.io.loadmat('CGH_tutorial/HW_58_pol_60_phase_correct.mat')
    totalSum = mat_data['totalSum']
    # Assume totalSum is a column/row vector; flip it vertically
    flipphase = np.flipud(totalSum).flatten()
    # Create ydata from 255 to 0 in 256 steps
    ydata = np.linspace(255, 0, 256)
    # Get unique values (sorted) and the indices
    phaseVal, ia = np.unique(flipphase, return_index=True)
    SLMDigit = ydata[ia]
    # Center phaseVal to be in [-pi, pi]
    phaseCenter = (np.max(phaseVal) + np.min(phaseVal)) / 2
    phaseVal = phaseVal - phaseCenter
    return phaseVal, SLMDigit

def calculate_hologram(patterns, zs, res_X, res_Y):
    # Parameters
    width, height = res_X, res_Y

    # Rescale
    scaled_patterns = []
    for i in range(len(patterns)):
        scaled_patterns.append(scale_and_pad(patterns[i], (height, width), 0))

    scaled_patterns = np.array(scaled_patterns).transpose(1,2,0)
    # x and y arrays (MATLAB: 1:600 and 1:800) – we use 0-indexed values but it cancels in the mean.
    x = np.arange(1, width + 1)
    y = np.arange(1, height + 1)
    # Create meshgrid for 2D operations (y: vertical, x: horizontal)
    X, Y = np.meshgrid(x, y)
    # ramp = 2*pi*(0.5*(x+y)) => equivalent to np.pi*(X+Y)
    ramp = np.pi * (X + Y)
        
    # Get the SLM lookup table
    phaseVal, SLMDigit = SLM_LUT()
        
    # Optical parameters
    dx = 10e-6
    dy = dx
    wavelength = 532e-9
    f = 100e-3

    # Quadratic phase factors: define functions for clarity
    def quad_phase(z):
        # Compute the quadratic phase term.
        # Note: subtracting mean(x) and mean(y) centers the coordinates.
        phase_x = ((dx / (wavelength * f)) * (X - np.mean(x))) ** 2
        phase_y = ((dy / (wavelength * f)) * (Y - np.mean(y))) ** 2
        return np.exp(-1j * np.pi * wavelength * z * (phase_x + phase_y))

    # Pattern 1 (from cameraman.tif)
    # pattern1_img = np.fliplr(pattern1_img)
    # Apply FFT: ifftshift -> FFT2 -> fftshift
    combined = np.zeros((height, width), dtype=np.complex128)
    for i in range(scaled_patterns.shape[2]):
        # rand_phase = np.exp(1j * 2 * np.pi * np.random.rand(height, width))
        # pattern_complex = scaled_patterns[:,:,i] * rand_phase
        pattern_complex = scaled_patterns[:,:,i]
        pattern = fftshift(fft2(ifftshift(pattern_complex)))
                                
        pattern *= quad_phase(zs[i])  # z1 is zero so this is effectively 1
        # Combine patterns and extract phase
        combined += pattern

    pattern_phase = np.angle(combined)
        
    # Add ramp and wrap to [-pi, pi]
    wrapped_phase = wrapToPi(pattern_phase + ramp)
    # Map the phase using the lookup table with nearest neighbor interpolation.
    # Create an interpolation function.
    interp_func = interp1d(phaseVal, SLMDigit, kind='nearest', fill_value="extrapolate")
    mapped_pattern = interp_func(wrapped_phase)
    pattern_uint8 = np.uint8(mapped_pattern)

    return pattern_uint8