import cv2
import numpy as np
from skimage.restoration import denoise_bilateral
from skimage import img_as_ubyte

# Load the image in color
image = cv2.imread("input_1017.png")  # Replace with your image file

# Convert to float32 for processing (float for precision in calculations)
image_float = image.astype(np.float32)

# 1. Gamma Correction for brightness adjustment
gamma = 0.7  # Lower gamma makes the image brighter
gamma_corrected = np.power(image_float / 255.0, gamma) * 255.0
gamma_corrected = np.clip(gamma_corrected, 0, 255)

# 2. Exposure Fusion / Tone Mapping (Apply a simple tone mapping)
exposure_factor = 2  # Exposure scaling
tone_mapped_image = gamma_corrected * exposure_factor
tone_mapped_image = np.clip(tone_mapped_image, 0, 255)

# 3. Decrease Contrast (Scale pixel values towards the midpoint)
contrast_factor = 0.9  # Reduce contrast by scaling pixel values closer to the midpoint (128)
midpoint = 15
decreased_contrast_image = (tone_mapped_image - midpoint) * contrast_factor + midpoint
decreased_contrast_image = np.clip(decreased_contrast_image, 0, 255)

# Display the image
cv2.imshow("Decreased Contrast Image", decreased_contrast_image.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()
