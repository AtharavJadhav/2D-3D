import numpy as np
import cv2 as cv
import glob
import os

### FIND CHESSBOARD CORNERS: objPoints and imgPoints

chessBoardSize = (9, 6)
frameSize = (2592, 4608)

### Termination Criteria

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

### Prepare object Points
objp = np.zeros((chessBoardSize[0] * chessBoardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessBoardSize[0], 0:chessBoardSize[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objPoints = []  # 3D point in real-world space
imgPoints = []  # 2D point in image plane

source_path_1 = "C:/Users/Atharav Jadhav/source/repos/2D-3D/Calibration Images"
images = [os.path.join(source_path_1, f).replace("\\", "/") for f in glob.glob(os.path.join(source_path_1, '*.jpg'))]

for image in images:
    print(image)
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, chessBoardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objPoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgPoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessBoardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)

cv.destroyAllWindows()

# Calibration
if len(objPoints) > 0 and len(imgPoints) > 0:
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, frameSize, None, None)

    if ret:
        # Save camera matrix and distortion coefficients as .npy files
        np.save("camera_matrix.npy", cameraMatrix)
        np.save("distortion_coeffs.npy", dist)
        print("Camera calibration parameters saved.")

        print("Camera Calibrated: ", ret)
        print("\nCamera Matrix:\n", cameraMatrix)
        print("\nDistortion Parameters:\n", dist)
        print("\nRotation Vectors:\n", rvecs)
        print("\nTranslation Vectors:\n", tvecs)
    else:
        print("Failed to calibrate the camera.")

else:
    print("Insufficient data for calibration.")


#### Code for Undistortion: To be added later

