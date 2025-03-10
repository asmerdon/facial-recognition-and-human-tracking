import cv2
import numpy as np
from ultralytics import YOLO
from moviepy.editor import VideoFileClip

# Load YOLOv8-Seg model (segmentation instead of detection)
model = YOLO("yolov8n-seg.pt")

def detect_people(frame):
    """ Detect people using YOLOv8 segmentation and return masks instead of bounding boxes. """
    results = model(frame)
    masks = []
    for result in results:
        for mask in result.masks.xy:
            masks.append(np.array(mask, np.int32))  # Convert polygon to NumPy array
    return masks

def generate_static_effect(frame, mask):
    """Generate static/noise effect to replace detected people."""
    noise = np.random.randint(0, 256, frame.shape, dtype=np.uint8)  # Generate random noise
    mask_fill = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_fill, [mask], 255)
    static_effect = cv2.bitwise_and(noise, noise, mask=mask_fill)  # Apply noise within mask
    frame = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask_fill))  # Remove detected people
    frame += static_effect  # Blend static effect into frame
    return frame

def process_frame(frame):
    """ Process a single frame: replace detected people with static effect."""
    masks = detect_people(frame)
    for mask in masks:
        frame = generate_static_effect(frame, mask)
    return frame

def process_video(input_path, output_path, process_full_video=True):
    """ Process an entire video and save the output. Can limit processing to 30 seconds. """
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if not process_full_video:
        max_frames = min(total_frames, fps * 30)  # Process only 30 seconds
    else:
        max_frames = total_frames
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        out.write(processed_frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    print(f"Processing complete! Total frames processed: {frame_count}")

# Run the script on a sample video
process_video("input.mp4", "output.mp4", process_full_video=False)  # Change to True for full video
