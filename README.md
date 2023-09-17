# 2D-3D Reconstruction Project

## Overview

This project aims to perform 2D to 3D reconstruction using computer vision techniques. It involves various steps such as camera calibration, feature extraction, sparse reconstruction, and bundle adjustment.

## Files

- [`_2D_3D.py`](https://github.com/AtharavJadhav/2D-3D/blob/main/_2D_3D.py): This file contains the code for camera calibration using chessboard images.
- [`Feature_Extraction.py`](https://github.com/AtharavJadhav/2D-3D/blob/main/Feature_Extraction.py): This file is responsible for extracting features from object images.
- [`Sparse_Reconstruction.py`](https://github.com/AtharavJadhav/2D-3D/blob/main/Sparse_Reconstruction.py): This file performs sparse reconstruction to generate 3D points.
- [`BundleAdjustment.py`](https://github.com/AtharavJadhav/2D-3D/blob/main/BundleAdjustment.py): This file optimizes the camera poses and 3D points using bundle adjustment.

## Dependencies

- OpenCV
- NumPy
- OpenSfM

## How to Run

1. Place your chessboard images in the `Calibration Images` folder.
2. Place your object images in the `Object Images` folder.
3. Run `_2D_3D.py` for camera calibration.
4. Run `Feature_Extraction.py` to extract features from object images.
5. Run `Sparse_Reconstruction.py` for sparse reconstruction.
6. Run `BundleAdjustment.py` for bundle adjustment.

## License

This project is open-source and available under the MIT License.

## Contributing

Feel free to contribute to this project by opening issues or submitting pull requests.

