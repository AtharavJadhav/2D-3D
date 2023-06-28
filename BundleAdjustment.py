import numpy as np
import cv2
import json
import opensfm

# Load camera parameters
camera_matrix = np.load("camera_matrix.npy")
distortion_coeffs = np.load("distortion_coeffs.npy")

# Load image paths
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

# Initialize OpenSFM reconstruction object
reconstruction = opensfm.Reconstruction()

# Loop through the images and add them to the reconstruction
for i, image_path in enumerate(image_paths):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Undistort the image using camera parameters
    undistorted_image = cv2.undistort(gray, camera_matrix, distortion_coeffs)

    # Extract features (keypoints and descriptors)
    detector = cv2.ORB_create()
    keypoints, descriptors = detector.detectAndCompute(undistorted_image, None)

    # Convert keypoints and descriptors to OpenSFM format
    features = opensfm.Features()
    for kp, desc in zip(keypoints, descriptors):
        feature = opensfm.Feature(kp.pt[0], kp.pt[1], desc.tolist())
        features.append(feature)

    # Add the image and features to the reconstruction
    shot = opensfm.Shot()
    shot.id = f"image_{i+1}"
    shot.camera = opensfm.Camera()
    shot.camera.id = "camera"
    shot.camera.width = image.shape[1]
    shot.camera.height = image.shape[0]
    shot.camera.k1 = distortion_coeffs[0]
    shot.camera.k2 = distortion_coeffs[1]
    shot.camera.focal = camera_matrix[0, 0]
    shot.camera.px = camera_matrix[0, 2]
    shot.camera.py = camera_matrix[1, 2]
    shot.camera.dataset_type = "multiview"

    reconstruction.add_camera(shot.camera)
    reconstruction.add_shot(shot)
    reconstruction.add_features(shot.id, features)

# Perform bundle adjustment
bundle_adjustment_options = opensfm.BundleAdjustmentOptions()
bundle_adjustment_options.loss = "SoftLOneLoss"
bundle_adjustment_options.min_num_obs = 2
bundle_adjustment_options.retriangulate_all = True
bundle_adjustment_options.max_gcp_distance = 100.0

reconstruction_bundle_adjustment = opensfm.BundleAdjustment()
reconstruction_bundle_adjustment.set_options(bundle_adjustment_options)
reconstruction_bundle_adjustment.bundle(reconstruction)

# Get optimized camera poses and 3D points
optimized_camera_poses = reconstruction.get_camera_positions()
optimized_3d_points = reconstruction.points

# Print optimized camera poses and 3D points
for camera_id, camera_pose in optimized_camera_poses.items():
    print(f"Camera {camera_id}: {camera_pose}")

for point_id, point in optimized_3d_points.items():
    print(f"Point {point_id}: {point}")

# Save the reconstruction as OpenSFM dataset
opensfm.save_reconstruction(reconstruction, "reconstruction.json")
