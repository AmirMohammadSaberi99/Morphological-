# erosion_demo.py

import cv2
import numpy as np
import argparse

def apply_erosion(img, kernel_size=(3,3), iterations=1):
    """
    Apply morphological erosion to a binary image.
    
    Args:
        img           : Input binary image (0 or 255).
        kernel_size   : Size of the structuring element.
        iterations    : Number of erosion iterations.
    Returns:
        eroded_img    : Result after erosion.
    """
    # Create a rectangular structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    eroded = cv2.erode(img, kernel, iterations=iterations)
    return eroded

def main(input_path, kernel_size, iterations):
    # 1) Load image in grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {input_path}")
    
    # 2) Threshold to binary (if not already binary)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # 3) Erode
    eroded = apply_erosion(binary, kernel_size=kernel_size, iterations=iterations)
    
    # 4) Stack for display
    combined = np.hstack([
        cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
    ])
    
    # 5) Show
    cv2.imshow(f"Original (L) vs. Eroded (R) — Kernel {kernel_size}, Iter {iterations}", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply erosion on a binary image and visualize.")
    parser.add_argument("image", help="Path to input image (grayscale or binary).")
    parser.add_argument("--kernel", "-k", type=int, nargs=2, default=(3,3),
                        help="Structuring element size, e.g. --kernel 5 5")
    parser.add_argument("--iter", "-n", type=int, default=1,
                        help="Number of erosion iterations.")
    args = parser.parse_args()
    
    main(args.image, tuple(args.kernel), args.iter)
