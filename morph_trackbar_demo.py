# morph_trackbar_demo.py

"""
Interactive demo: adjust kernel size and operation for morphological transforms
using OpenCV trackbars.
"""

import cv2
import numpy as np
import argparse

# Mapping of trackbar values to operations
OPS = {
    0: cv2.MORPH_ERODE,
    1: cv2.MORPH_DILATE,
    2: cv2.MORPH_OPEN,
    3: cv2.MORPH_CLOSE
}
OP_NAMES = {
    0: "Erosion",
    1: "Dilation",
    2: "Opening",
    3: "Closing"
}

def apply_morph(img_bin, op, kx, ky):
    """Apply the specified morphological operation."""
    # enforce odd kernel sizes >= 1
    kx = max(1, kx // 2 * 2 + 1)
    ky = max(1, ky // 2 * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    return cv2.morphologyEx(img_bin, op, kernel)

def on_trackbar(_):
    """Callback (no-op) required by createTrackbar."""
    pass

def main(image_path):
    # Load and threshold
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    _, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # Create window and trackbars
    win_name = "Morphology Demo"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Trackbar for operation type
    cv2.createTrackbar("Op: 0=Erode,1=Dilate,2=Open,3=Close",
                       win_name, 0, len(OPS)-1, on_trackbar)
    # Trackbars for kernel size
    cv2.createTrackbar("Kernel X", win_name, 3, 31, on_trackbar)
    cv2.createTrackbar("Kernel Y", win_name, 3, 31, on_trackbar)

    while True:
        # Read trackbar positions
        op_idx = cv2.getTrackbarPos("Op: 0=Erode,1=Dilate,2=Open,3=Close", win_name)
        kx = cv2.getTrackbarPos("Kernel X", win_name)
        ky = cv2.getTrackbarPos("Kernel Y", win_name)

        # Apply morphological operation
        result = apply_morph(img_bin, OPS[op_idx], kx, ky)

        # Stack original & result
        orig_bgr = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
        res_bgr  = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        # Annotate
        cv2.putText(orig_bgr, "Original", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        label = f"{OP_NAMES[op_idx]} (kx={kx}, ky={ky})"
        cv2.putText(res_bgr, label, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        combined = np.hstack([orig_bgr, res_bgr])
        cv2.imshow(win_name, combined)

        key = cv2.waitKey(50) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive morphological operations with adjustable kernel"
    )
    parser.add_argument(
        "image",
        help="Path to a grayscale or binary image to process"
    )
    args = parser.parse_args()
    main(args.image)
