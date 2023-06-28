#### Feature Extraction and Matching

import cv2
import numpy as np
import json

# Load the camera parameters
camera_matrix = np.load("camera_matrix.npy")
distortion_coeffs = np.load("distortion_coeffs.npy")

# Load the paths to the object images
image_paths = [
    "C:/Users/Atharav Jadhav/source/repos/2D-3D/Object Images/img1.jpg",
    "C:/Users/Atharav Jadhav/source/repos/2D-3D/Object Images/img2.jpg",
    "C:/Users/Atharav Jadhav/source/repos/2D-3D/Object Images/img3.jpg",
    "C:/Users/Atharav Jadhav/source/repos/2D-3D/Object Images/img4.jpg",
    "C:/Users/Atharav Jadhav/source/repos/2D-3D/Object Images/img5.jpg",
    "C:/Users/Atharav Jadhav/source/repos/2D-3D/Object Images/img6.jpg",
    "C:/Users/Atharav Jadhav/source/repos/2D-3D/Object Images/img7.jpg"
    # Add paths for the rest of the images
]

# Create feature detector and descriptor extractor
detector = cv2.ORB_create()
descriptor = cv2.ORB_create()

# Initialize lists to store keypoints and descriptors
keypoints_list = []
descriptors_list = []

# Loop through the images and extract features
for i, image_path in enumerate(image_paths):
    print(f"Processing image {i+1}/{len(image_paths)}")
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Undistort the image using camera parameters
    undistorted_image = cv2.undistort(gray, camera_matrix, distortion_coeffs)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = detector.detectAndCompute(undistorted_image, None)
    print(f"Number of keypoints detected in image {i+1}: {len(keypoints)}")

    # Store the keypoints and descriptors
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)

    # Save keypoints as a JSON file
    keypoints_file = f"keypoints_{i+1}.json"
    keypoints_data = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

    with open(keypoints_file, "w") as keypoints_output:
        json.dump(keypoints_data, keypoints_output)

    print("Keypoints saved as", keypoints_file)

    # Save descriptors as a JSON file
    descriptors_file = f"descriptors_{i+1}.json"
    descriptors_data = descriptors.tolist()

    with open(descriptors_file, "w") as descriptors_output:
        json.dump(descriptors_data, descriptors_output)

    print("Descriptors saved as", descriptors_file)

# Initialize list to store matches
matches_list = []

# Loop through pairs of images
for i in range(len(image_paths) - 1):
    print(f"Matching features between images {i+1} and {i+2}")
    # Match features between the current image and the next image
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_list[i], descriptors_list[i+1])
    print(f"Number of matches between images {i+1} and {i+2}: {len(matches)}")

    # Store the matches
    matches_list.append(matches)

# Save matches as a JSON file
matches_file = "matches.json"
matches_data = []
for matches in matches_list:
    matches_data.append([(match.queryIdx, match.trainIdx) for match in matches])

with open(matches_file, "w") as matches_output:
    json.dump(matches_data, matches_output)

print("Matches saved as", matches_file)