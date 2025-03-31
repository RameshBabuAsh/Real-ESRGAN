import cv2
import numpy as np

def reduce_noise_bilateral(image, d=9, sigmaColor=50, sigmaSpace=50):
    # Bilateral filtering tends to preserve edges while smoothing homogeneous regions.
    denoised = cv2.bilateralFilter(image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return denoised

def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def adjust_gamma(frame, gamma=1.7):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(frame, table)

def adjust_contrast(frame, alpha=1.5, beta=1):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image")
        return

    # Step 1: Use bilateral filter to reduce noise while preserving details.
    
    # Step 2: Apply enhancement steps.
    enhanced = adjust_gamma(image, gamma=1.7)
    enhanced = apply_clahe(enhanced)
    enhanced = adjust_contrast(enhanced, alpha=1.5, beta=1)
    enhanced = reduce_noise_bilateral(enhanced, d=9, sigmaColor=50, sigmaSpace=50)

    combined = np.hstack((image, enhanced))
    cv2.imshow("Original (left) vs Enhanced (right)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "low.png"  # replace with your image file path
    process_image(image_path)
