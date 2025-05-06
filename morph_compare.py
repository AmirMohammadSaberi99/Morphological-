# morph_compare.py

import cv2
import numpy as np
import argparse

def apply_morphology(img_bin, op, kernel_size=(3,3), iterations=1):
    """
    Apply a morphological operation (erosion or dilation) to a binary image.
    
    Args:
        img_bin      : Input binary image (0 or 255).
        op           : cv2.MORPH_ERODE or cv2.MORPH_DILATE.
        kernel_size  : Size of the structuring element, e.g. (5,5).
        iterations   : Number of times to apply the operation.
    Returns:
        result_img   : Resulting binary image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(img_bin, op, kernel, iterations=iterations)

def main(image_path, kx, ky, iters):
    # 1) Load in grayscale
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    
    # 2) Threshold to binary
    _, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    
    # 3) Apply erosion and dilation
    kernel_size = (kx, ky)
    eroded   = apply_morphology(img_bin, cv2.MORPH_ERODE,   kernel_size, iterations=iters)
    dilated  = apply_morphology(img_bin, cv2.MORPH_DILATE,  kernel_size, iterations=iters)
    
    # 4) Stack for display: Original | Eroded | Dilated
    # Convert each to BGR so we can draw colored labels
    orig_bgr   = cv2.cvtColor(img_bin,  cv2.COLOR_GRAY2BGR)
    erode_bgr  = cv2.cvtColor(eroded,    cv2.COLOR_GRAY2BGR)
    dilate_bgr = cv2.cvtColor(dilated,   cv2.COLOR_GRAY2BGR)
    
    # Annotate
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(orig_bgr,   'Original', (10,30), font, 1, (255,255,255), 2)
    cv2.putText(erode_bgr,  'Erosion',  (10,30), font, 1, (0,0,255),    2)
    cv2.putText(dilate_bgr, 'Dilation', (10,30), font, 1, (0,255,0),    2)
    
    combined = np.hstack([orig_bgr, erode_bgr, dilate_bgr])
    
    # 5) Display
    window_name = f"Kernel={kernel_size}, Iter={iters}"
    cv2.imshow(window_name, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare erosion vs. dilation on a binary image"
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
        help="Number of times to apply each operation"
    )
    args = parser.parse_args()
    
    main(args.image, args.kernel[0], args.kernel[1], args.iterations)
