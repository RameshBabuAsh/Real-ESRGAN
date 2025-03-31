import cv2
import numpy as np

def reduce_noise_bilateral(image, d=9, sigmaColor=50, sigmaSpace=50):
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

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

def process_video(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (360, 360))
        
        enhanced = adjust_gamma(frame, gamma=1.7)
        enhanced = apply_clahe(enhanced)
        enhanced = adjust_contrast(enhanced, alpha=1.5, beta=1)
        enhanced = reduce_noise_bilateral(enhanced, d=9, sigmaColor=50, sigmaSpace=50)
        
        combined = np.hstack((frame, enhanced))
        cv2.imshow("Original (left) vs Enhanced (right)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    source = 0
    process_video(source)
