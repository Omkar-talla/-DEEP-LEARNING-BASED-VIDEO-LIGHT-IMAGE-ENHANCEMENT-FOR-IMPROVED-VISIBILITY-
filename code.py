import argparse
from argparse import RawTextHelpFormatter
import glob
import os
from os import makedirs
from os.path import join, exists, basename, splitext
import cv2
import numpy as np
from scipy import fft
from skimage import io, exposure, img_as_ubyte, img_as_float
from tqdm import trange
import matplotlib.pyplot as plt
from PIL import Image
import shutil
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from sklearn.metrics import mean_squared_error

# Function to enhance images using brightness, contrast, and gamma correction
def enhance_image_existing(input_path, output_path):
    brightness_alpha = 1.2
    contrast_beta = 10
    gamma = 1.5

    img = cv2.imread(input_path)

    # Apply brightness adjustment
    enhanced_brightness = cv2.convertScaleAbs(img, alpha=brightness_alpha, beta=0)

    # Apply contrast adjustment
    enhanced_contrast = cv2.addWeighted(enhanced_brightness, 1, img, 0, contrast_beta)

    # Apply gamma correction
    enhanced_img = np.power(enhanced_contrast / 255.0, gamma)
    enhanced_img = np.uint8(enhanced_img * 255)

    # Save the enhanced image
    cv2.imwrite(output_path, enhanced_img)

    return enhanced_img

# Function to enhance image using LIME (Local Image Manipulation Enhancement)
def enhance_image_with_lime(input_path, output_path):
    image = cv2.imread(input_path)
    # Assuming `enhance_image_exposure` is a custom function that applies LIME
    enhanced_image = enhance_image_exposure(image, 0.6, 0.15, "lime")

    # Save the enhanced image
    cv2.imwrite(output_path, enhanced_image)

    return enhanced_image

# Function to compute PSNR, SSIM, and MSE between original and enhanced images
def compute_metrics(original, enhanced):
    psnr_value = psnr(original, enhanced)

    # Calculate SSIM with a suitable window size
    win_size = 3
    ssim_value, _ = ssim(original, enhanced, full=True, win_size=win_size)

    mse_value = mean_squared_error(original.flatten(), enhanced.flatten())

    return psnr_value, ssim_value, mse_value

# Video Processing (Reading, Enhancing, and Saving Video)
input_video_path = 'Input Videos/Amazing night vision - ColorVu Camera Demo.mp4'
input_frames_folder = 'input_frames/'
output_fps = 5  # frames per second
duration = 5  # seconds

output_frames_folder1 = 'output_frames_existing/'
output_video_path_existing = 'output_video_existing.avi'
output_frames_folder = 'output_frames/'
output_video_path_proposed = 'output_video_proposed.avi'

# Open the video file
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_frames = output_fps * duration

# Create a video writer object for existing method
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path_existing, fourcc, 25.0, (frame_width, frame_height))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret or frame_count >= num_frames:
        break

    # Save the input frame
    input_frame_path = os.path.join(input_frames_folder, f"input_frame_{frame_count + 1}.png")
    frame = cv2.resize(frame, (400, 400))  # Resize for processing
    cv2.imwrite(input_frame_path, frame)

    # Enhance the image using existing method
    output_frame_path = os.path.join(output_frames_folder1, f"output_frame_{frame_count + 1}.png")
    enhance_image_existing(input_frame_path, output_frame_path)

    # Write the output frame to the video
    out.write(cv2.imread(output_frame_path))

    frame_count += 1
    print(f"Output Frames Completed for Existing Method: {frame_count}")

# Release video capture and writer objects
cap.release()
out.release()
print(f"Processing completed. Existing output video created and stored at {output_video_path_existing}")

# Process video using LIME enhancement
cap = cv2.VideoCapture(input_video_path)
out = cv2.VideoWriter(output_video_path_proposed, fourcc, 25.0, (frame_width, frame_height))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret or frame_count >= num_frames:
        break

    # Save the input frame
    input_frame_path = os.path.join(input_frames_folder, f"input_frame_{frame_count + 1}.png")
    frame = cv2.resize(frame, (400, 400))  # Resize for processing
    cv2.imwrite(input_frame_path, frame)

    # Enhance the image using LIME
    output_frame_path = os.path.join(output_frames_folder, f"output_frame_{frame_count + 1}.png")
    enhance_image_with_lime(input_frame_path, output_frame_path)

    # Write the output frame to the video
    out.write(cv2.imread(output_frame_path))

    frame_count += 1
    print(f"Output Frames Completed for Proposed Method: {frame_count}")

# Release video capture and writer objects
cap.release()
out.release()
print(f"Processing completed. Proposed output video created and stored at {output_video_path_proposed}")

# Load the original image
img = cv2.imread('sample images/15.jpg')

# Existing enhancement
enhanced_img1 = enhance_image_existing('sample images/15.jpg', 'enhanced_existing.jpg')

# Proposed enhancement (LIME)
enhanced_img2 = enhance_image_with_lime('sample images/15.jpg', 'enhanced_proposed.jpg')

# Compute metrics for the existing and proposed methods
psnr_existing, ssim_existing, mse_existing = compute_metrics(img, enhanced_img1)
psnr_proposed, ssim_proposed, mse_proposed = compute_metrics(img, enhanced_img2)

# Print results
print("Metrics for Existing Enhancement:")
print(f"PSNR: {psnr_existing}")
print(f"SSIM: {ssim_existing}")
print(f"MSE: {mse_existing}")

print("\nMetrics for Proposed Enhancement:")
print(f"PSNR: {psnr_proposed}")
print(f"SSIM: {ssim_proposed}")
print(f"MSE: {mse_proposed}")
