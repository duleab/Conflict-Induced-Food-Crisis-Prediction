import cv2
import os
import sys

def compress_video(input_path, output_path, target_width=854): # 480p width
    """
    Compresses a video by resizing to 480p.
    Using OpenCV which doesn't have great bitrate control, resizing 
    is the most reliable way to drop file size.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate scale for 480p
    scale = target_width / width
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    print(f"Original: {width}x{height}, {fps} FPS")
    print(f"Targeting: {new_width}x{new_height}")
    sys.stdout.flush()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (new_width, new_height))
        out.write(frame)
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Progress: {frame_count}/{total_frames}")
            sys.stdout.flush()

    cap.release()
    out.release()
    
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Finished! New Size: {new_size:.2f} MB")
    sys.stdout.flush()

if __name__ == "__main__":
    input_file = "images/Conflict-Induced-Food-Crisis-Prediction01.mp4"
    output_file = "images/Conflict-Induced-Food-Crisis-Prediction01_compressed.mp4"
    compress_video(input_file, output_file)
