import cv2
import numpy as np

def compute_vlight_LUT(v):
    """
    Compute the LUT for VLight enhancement based on parameter v.
    Input:
      v: enhancement parameter in [0, 1). When v is near 0, little enhancement is applied.
    Output:
      lut: a 256-element uint8 numpy array mapping input intensity to output.
    """
    # Calculate b(v) and G(v)
    b = 1.0 / (5.0 * v + 0.05)  # Avoid division by zero when v=0 (v is float; if v==0, b becomes 20)
    G = 1.0 - v * v

    # For input values in [0, 1], define the transfer function f(x) = arctan(-G*(1 + b*x))
    # We then normalize f(x) to the range [0, 1] by using its values at x=0 and x=1.
    f0 = np.arctan(-G * (1.0))        # when x = 0
    f1 = np.arctan(-G * (1.0 + b))      # when x = 1

    # Build LUT: for each input intensity I (0..255), map to normalized output.
    lut = np.zeros(256, dtype=np.uint8)
    for I in range(256):
        x = I / 255.0  # normalize input to [0,1]
        f = np.arctan(-G * (1.0 + b * x))
        # Normalize: since f0 and f1 are both negative and f is monotonic,
        # map f = f0 to 0 and f = f1 to 1.
        mapped = (f - f0) / (f1 - f0)
        # Scale to 0-255 and clip
        lut[I] = np.clip(mapped * 255, 0, 255)
    return lut

def adapt_parameter(v_channel, Eth=0.4):
    """
    Adaptively compute the enhancement parameter v based on the average brightness.
    v = 1 - min(Eavg / Eth, 1)
    where Eavg is the average pixel intensity in [0, 1].
    """
    Eavg = np.mean(v_channel / 255.0)
    v = 1.0 - min(Eavg / Eth, 1.0)
    return v

def process_frame(frame, Eth=0.4):
    """
    Process a single frame using VLight.
      - Converts frame to HSV
      - Adapts enhancement parameter v using the V channel
      - Computes LUT for VLight and applies it to the V channel
      - Converts back to BGR for display
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # Compute adaptive parameter v
    v_param = adapt_parameter(V, Eth=Eth)

    # Compute LUT using current v. (Optionally, you might cache the LUT if v changes slowly.)
    lut = compute_vlight_LUT(v_param)

    # Apply LUT to the V channel
    V_enhanced = cv2.LUT(V, lut)

    # Merge channels and convert back to BGR
    hsv_enhanced = cv2.merge([H, S, V_enhanced])
    enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    return enhanced, v_param

def process_video(video_source=0, Eth=0.4):
    """
    Process a video stream (from a file or camera) applying VLight enhancement in real time.
      video_source: path to video file or camera index (default 0).
      Eth: threshold for adapting parameter v.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error opening video source")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optionally, resize frame for faster processing (e.g., for Raspberry Pi)
        frame = cv2.resize(frame, (640, 480))
        enhanced_frame, v_param = process_frame(frame, Eth=Eth)

        # Display original and enhanced frames side by side
        combined = np.hstack((frame, enhanced_frame))
        cv2.putText(combined, f"v = {v_param:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Original (left) vs Enhanced (right)", combined)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # To test with a video file, replace 0 with the filename, e.g., "input_video.mp4"
    process_video(video_source="sample_dark.webm", Eth=0.1)
