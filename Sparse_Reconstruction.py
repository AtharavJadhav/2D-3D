import numpy as np
import cv2

camera_matrix = np.load("camera_matrix.npy")
distortion_coeffs = np.load("distortion_coeffs.npy")

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

import json

# Load the matches from the JSON file
with open("matches.json", "r") as matches_file:
    matches_data = json.load(matches_file)

# Extract the matches for each image pair
matches_list = []
for matches in matches_data:
    matches_list.append([cv2.DMatch(match[0], match[1], 0) for match in matches])

# Load the keypoints and descriptors for each image
keypoints_list = []
descriptors_list = []

for i, image_path in enumerate(image_paths):
    keypoints_file = f"keypoints_{i+1}.json"
    descriptors_file = f"descriptors_{i+1}.json"

    with open(keypoints_file, "r") as keypoints_input:
        keypoints_data = json.load(keypoints_input)
        keypoints = []
        for kp in keypoints_data:
            x, y = kp[0]
            size = float(kp[1])
            angle = float(kp[2])
            keypoints.append(cv2.KeyPoint(x, y, size, angle))

    with open(descriptors_file, "r") as descriptors_input:
        descriptors_data = json.load(descriptors_input)
        descriptors = np.array(descriptors_data)

    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)

from itertools import combinations

# Initialize the list to store camera poses
camera_poses = []

# Loop through image pairs
for i in range(len(image_paths) - 1):
    # Get the matches for the current image pair
    matches = matches_list[i]

    # Get the keypoints for the current image pair
    keypoints1 = keypoints_list[i]
    keypoints2 = keypoints_list[i+1]

    # Extract the matching keypoints
    src_pts = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

    # Estimate the fundamental matrix using RANSAC
    fundamental_matrix, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 3.0, 0.99)

    # Select the inlier matches
    src_pts = src_pts[mask.ravel() == 1]
    dst_pts = dst_pts[mask.ravel() == 1]

    # Estimate the essential matrix from the fundamental matrix and camera matrix
    essential_matrix, _ = cv2.findEssentialMat(src_pts, dst_pts, camera_matrix)

    # Recover the relative camera poses
    _, rotation, translation, _ = cv2.recoverPose(essential_matrix, src_pts, dst_pts, camera_matrix)

    # Create the camera pose matrix
    pose = np.hstack((rotation, translation))

    # Store the camera pose
    camera_poses.append(pose)


# Convert the camera_poses list to a regular Python list
camera_poses_list = [pose.tolist() for pose in camera_poses]
# Save the camera poses
with open("camera_poses.json", "w") as camera_poses_file:
    json.dump(camera_poses_list, camera_poses_file)

# Initialize the list to store the triangulated 3D points
triangulated_points = []

# Loop through image pairs
for i in range(len(image_paths) - 2):
    # Get the matches for the current image pair
    matches = matches_list[i]

    # Get the keypoints for the current image pair
    keypoints1 = keypoints_list[i]
    keypoints2 = keypoints_list[i+1]

    # Get the camera poses for the current image pair
    pose1 = camera_poses[i]
    pose2 = camera_poses[i+1]

    # Extract the coordinates from keypoints1 and keypoints2
    keypoints1_coords = np.float32([kp.pt for kp in keypoints1]).reshape(-1, 1, 2)
    keypoints2_coords = np.float32([kp.pt for kp in keypoints2]).reshape(-1, 1, 2)

    # Triangulate 3D points
    points_4d_homogeneous = cv2.triangulatePoints(camera_matrix @ pose1, camera_matrix @ pose2, keypoints1_coords, keypoints2_coords)
    points_3d_homogeneous = points_4d_homogeneous / points_4d_homogeneous[3]
    points_3d = points_3d_homogeneous[:3].T

    # Store the triangulated points
    triangulated_points.append(points_3d)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Plot the 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Loop through triangulated points
for points in triangulated_points:
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')

# Set plot parameters
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Reconstruction')

# Show the plot
plt.show()