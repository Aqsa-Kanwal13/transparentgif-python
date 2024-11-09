# import os
# import cv2
# import numpy as np
# from rembg import remove
# from PIL import Image
# import time

# # Extract 50 frames from the video
# def extract_frames(video_path, frame_folder):
#     if not os.path.exists(frame_folder):
#         os.makedirs(frame_folder)

#     vidcap = cv2.VideoCapture(video_path)
#     if not vidcap.isOpened():
#         print(f"Error: Could not open video file {video_path}")
#         return

#     original_fps = vidcap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"Original video FPS: {original_fps}")
#     print(f"Total frames in video: {total_frames}")

#     max_frames = 50  # Set the max frames to extract
#     saved_frames = 0  # Counter for saved frames

#     start_time = time.time()  # Start timing the extraction process

#     while saved_frames < max_frames:
#         success, image = vidcap.read()  # Read the next frame
#         if success:
#             # Convert BGR to RGB (as rembg works with RGB format)
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             # Remove background using rembg
#             image_rgb_pil = Image.fromarray(image_rgb)
#             output = remove(image_rgb_pil)

#             # Convert RGBA back to OpenCV format for saving
#             output_np = np.array(output)

#             # Save the frame with transparent background as PNG
#             frame_path = os.path.join(frame_folder, f"frame_{saved_frames:04d}.png")
#             Image.fromarray(output_np).save(frame_path)
#             saved_frames += 1  # Increment the saved frames counter
#         else:
#             print(f"Warning: Failed to read frame.")
#             break

#     vidcap.release()
#     extraction_time = time.time() - start_time  # Calculate extraction time
#     print(f"Extracted {saved_frames} frames with transparent background in {extraction_time:.2f} seconds.")
#     return original_fps, saved_frames, extraction_time

# # Create a transparent GIF from the extracted frames
# def create_gif_with_transparency(frame_folder, gif_path, duration, size=(512, 512)):
#     frame_files = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.png')])

#     if not frame_files:
#         print("No frames found to create GIF.")
#         return

#     frames = []
#     start_time = time.time()  # Start timing the GIF creation process
#     for frame_file in frame_files:
#         frame = Image.open(frame_file).convert("RGBA")  # Ensure RGBA format
#         frame = frame.resize(size, Image.LANCZOS)  # Resize to 512x512
#         frames.append(frame)

#     try:
#         # Prepare a new list to convert each RGBA frame to a P palette mode (to respect transparency)
#         palette_frames = []
#         for frame in frames:
#             # Convert RGBA to P (with transparency)
#             frame_p = frame.convert("P", palette=Image.ADAPTIVE, colors=256)

#             # Set the transparency index by finding the alpha channel index
#             alpha = frame.getchannel('A')
#             mask = Image.eval(alpha, lambda a: 255 if a <= 128 else 0)
#             frame_p.paste(255, mask)  # Ensure transparent parts are handled

#             palette_frames.append(frame_p)

#         # Save as transparent GIF
#         palette_frames[0].save(
#             gif_path,
#             save_all=True,
#             append_images=palette_frames[1:],
#             optimize=False,
#             duration=duration,
#             loop=0,
#             transparency=255,  # This ensures transparency is respected
#             disposal=2         # Clears each frame before the next one
#         )
#         gif_creation_time = time.time() - start_time  # Calculate GIF creation time
#         print(f"GIF created with transparency and saved to {gif_path} in {gif_creation_time:.2f} seconds.")
#     except Exception as e:
#         print(f"Error creating GIF: {e}")

# # Parameters
# video_path = 'C:/Users/IDEAL/Desktop/transparentgif/abctest.mp4'  # Use the path of your uploaded video
# frame_folder = 'extracted_frames'
# gif_path = 'output_transparent.gif'
# duration = 50  # Duration in ms per frame (you can adjust this)

# # Extract frames from the video with background removal
# original_fps, saved_frames, extraction_time = extract_frames(video_path, frame_folder)

# # Create a transparent GIF from the extracted frames
# create_gif_with_transparency(frame_folder, gif_path, duration)

# # Print final report
# print(f"Original video FPS: {original_fps}")
# print(f"GIF FPS: {1000 / duration}") 
# print(f"Total extraction time: {extraction_time:.2f} seconds.")
import os
import cv2
import numpy as np
from rembg import remove
from PIL import Image
import time
from concurrent.futures import ThreadPoolExecutor

# Function to process a single frame
def process_frame(image):
    # Convert BGR to RGB (as rembg works with RGB format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Remove background using rembg
    image_rgb_pil = Image.fromarray(image_rgb)
    output = remove(image_rgb_pil)

    # Convert RGBA back to OpenCV format for saving
    output_np = np.array(output)
    return output_np

# Extract 50 frames from the video
def extract_frames(video_path, frame_folder):
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    original_fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Original video FPS: {original_fps}")
    print(f"Total frames in video: {total_frames}")

    max_frames = 50  # Set the max frames to extract
    saved_frames = 0  # Counter for saved frames
    frames_to_process = []

    while saved_frames < max_frames:
        success, image = vidcap.read()  # Read the next frame
        if success:
            frames_to_process.append(image)
            saved_frames += 1  # Increment the saved frames counter
        else:
            print(f"Warning: Failed to read frame.")
            break

    vidcap.release()

    # Process frames in parallel using ThreadPoolExecutor
    start_time = time.time()  # Start timing the extraction process

    if frames_to_process:
        cpu_count = os.cpu_count()  # Get the number of CPU cores
        with ThreadPoolExecutor(max_workers=cpu_count) as executor:  # Use all available cores
            results = list(executor.map(process_frame, frames_to_process))

        # Save the processed frames
        for idx, output_np in enumerate(results):
            frame_path = os.path.join(frame_folder, f"frame_{idx:04d}.png")
            Image.fromarray(output_np).save(frame_path)

    extraction_time = time.time() - start_time  # Calculate extraction time
    print(f"Extracted {saved_frames} frames with transparent background in {extraction_time:.2f} seconds.")
    return original_fps, saved_frames, extraction_time

# Create a transparent GIF from the extracted frames
def create_gif_with_transparency(frame_folder, gif_path, duration, size=(512, 512)):
    frame_files = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.png')])

    if not frame_files:
        print("No frames found to create GIF.")
        return

    frames = []
    start_time = time.time()  # Start timing the GIF creation process
    for frame_file in frame_files:
        frame = Image.open(frame_file).convert("RGBA")  # Ensure RGBA format
        frame = frame.resize(size, Image.LANCZOS)  # Resize to 512x512
        frames.append(frame)

    try:
        # Prepare a new list to convert each RGBA frame to a P palette mode (to respect transparency)
        palette_frames = []
        for frame in frames:
            # Convert RGBA to P (with transparency)
            frame_p = frame.convert("P", palette=Image.ADAPTIVE, colors=256)

            # Set the transparency index by finding the alpha channel index
            alpha = frame.getchannel('A')
            mask = Image.eval(alpha, lambda a: 255 if a <= 128 else 0)
            frame_p.paste(255, mask)  # Ensure transparent parts are handled

            palette_frames.append(frame_p)

        # Save as transparent GIF
        palette_frames[0].save(
            gif_path,
            save_all=True,
            append_images=palette_frames[1:],
            optimize=False,
            duration=duration,
            loop=0,
            transparency=255,  # This ensures transparency is respected
            disposal=2         # Clears each frame before the next one
        )
        gif_creation_time = time.time() - start_time  # Calculate GIF creation time
        print(f"GIF created with transparency and saved to {gif_path} in {gif_creation_time:.2f} seconds.")
    except Exception as e:
        print(f"Error creating GIF: {e}")

# Parameters
video_path = 'C:/Users/IDEAL/Desktop/transparentgif/media_20240926_090814_7419636361353733842.mp4'  # Use the path of your uploaded video
frame_folder = 'extracted_frames'
gif_path = 'output_transparent.gif'
duration = 50  # Duration in ms per frame (you can adjust this)

# Extract frames from the video with background removal
original_fps, saved_frames, extraction_time = extract_frames(video_path, frame_folder)

# Create a transparent GIF from the extracted frames
create_gif_with_transparency(frame_folder, gif_path, duration)

# Print final report
print(f"Original video FPS: {original_fps}")
print(f"GIF FPS: {1000 / duration}") 
print(f"Total extraction time: {extraction_time:.2f} seconds.")

# import shutil
# import os

# # Define the path to the folder
# folder_path = 'C:/Users/IDEAL/Desktop/transparentgif/extracted_frames'

# # Check if the folder exists
# if os.path.exists(folder_path):
#     # Remove the folder and its contents
#     shutil.rmtree(folder_path)
#     print(f"Folder '{folder_path}' and its contents have been deleted.")
# else:
#     print(f"Folder '{folder_path}' does not exist.")
