
# opening_demo.py

import cv2
import numpy as np
import argparse

def apply_opening(img_bin, kernel_size=(3,3), iterations=1):
    """
    Apply morphological opening (erosion followed by dilation) to a binary image.
    
    Args:
        img_bin     : Input binary image (0 or 255).
        kernel_size : Size of the structuring element, e.g. (3,3).
        iterations  : Number of times to apply the opening.
    Returns:
        opened_img  : Result after opening.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # Opening is just erosion followed by dilation
    opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return opened

def main(input_path, kx, ky, iterations):
    # 1) Load image in grayscale
    img_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Cannot load image: {input_path}")
    
    # 2) Threshold to binary (if not already binary)
    _, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    
    # 3) Apply opening to remove small noise
    kernel_size = (kx, ky)
    opened = apply_opening(img_bin, kernel_size, iterations)
    
    # 4) Stack for display: Original | Opened
    orig_bgr  = cv2.cvtColor(img_bin,  cv2.COLOR_GRAY2BGR)
    open_bgr  = cv2.cvtColor(opened,    cv2.COLOR_GRAY2BGR)
    cv2.putText(orig_bgr, "Original", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(open_bgr, "Opening", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    combined = np.hstack([orig_bgr, open_bgr])
    
    # 5) Display
    window_name = f"Opening: kernel={kernel_size}, iter={iterations}"
    cv2.imshow(window_name, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstrate morphological opening (erosion → dilation) on a binary image"
    )
    parser.add_argument(
        "image",
        help="Path to input image (grayscale or binary)"
    )
    parser.add_argument(
        "--kernel", "-k",
        type=int, nargs=2, default=[3,3],
        metavar=('KX','KY'),
        help="Structuring element size (width height), e.g. -k 5 5"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int, default=1,
        help="Number of times to apply opening"
    )
    args = parser.parse_args()
    
    main(args.image, args.kernel[0], args.kernel[1], args.iterations)
