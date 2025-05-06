# Morphological Operations Demo Toolkit

This repository contains a set of standalone Python scripts demonstrating basic morphological operations on binary images using OpenCV. You can apply erosion, dilation, opening, closing, and interactively adjust parameters via a trackbar GUI.

## Files

```
├── erosion_demo.py           # 1. Apply erosion on a binary image and visualize the result
├── morph_compare.py          # 2. Apply dilation on a binary image and compare with erosion
├── opening_demo.py           # 3. Use erosion followed by dilation (opening) to remove small noise
├── closing_demo.py           # 4. Use dilation followed by erosion (closing) to fill small holes
└── morph_trackbar_demo.py    # 5. Create a trackbar GUI to change kernel size for morphological operations
```

## Requirements

* Python 3.6+
* OpenCV Python

Install dependencies:

```bash
pip install opencv-python
```

## Usage

### 1. Erosion Demo

```bash
python erosion_demo.py <image_path> --kernel KX KY --iter N
```

* **`<image_path>`**: Path to input grayscale or binary image.
* **`--kernel KX KY`**: Structuring element size (default: 3 3).
* **`--iter N`**: Number of erosion iterations (default: 1).

Displays the original binary image on the left and the eroded result on the right.

### 2. Morphology Compare (Erosion vs. Dilation)

```bash
python morph_compare.py <image_path> --kernel KX KY --iterations N
```

* Compares original, eroded, and dilated images side by side with labels.

### 3. Opening Demo (Erosion → Dilation)

```bash
python opening_demo.py <image_path> --kernel KX KY --iterations N
```

* Removes small foreground noise via an opening operation.
* Displays original and opened images side by side.

### 4. Closing Demo (Dilation → Erosion)

```bash
python closing_demo.py <image_path> --kernel KX KY --iterations N
```

* Fills small background holes via a closing operation.
* Displays original and closed images side by side.

### 5. Interactive Morphological Trackbar GUI

```bash
python morph_trackbar_demo.py <image_path>
```

* Opens a GUI window with trackbars to select:

  * Operation: Erosion, Dilation, Opening, Closing
  * Kernel X size (1–31, odd enforced)
  * Kernel Y size (1–31, odd enforced)
* Displays the original binary image and the transformed result side by side in real time.
* Press **ESC** or **q** to exit.

## Notes

* All scripts threshold the input to a binary image using a fixed 127 threshold. Ensure your image has sufficient contrast for meaningful results.
* Adjust kernel sizes and iteration counts to see their effect on the morphological outcome.

---

© 2025 Morphology Demo Toolkit | MIT License
