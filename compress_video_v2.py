import imageio
import os
import sys

def compress_video_imageio(input_path, output_path, bitrate="800k"):
    """
    Compresses video using imageio and ffmpeg.
    Bitrate '800k' should keep a 2-minute video under 10MB easily.
    """
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print(f"Reading {input_path}...")
    try:
        reader = imageio.get_reader(input_path)
        fps = reader.get_meta_data()['fps']
        
        print(f"Original FPS: {fps}")
        print(f"Compressing with bitrate {bitrate}...")
        
        writer = imageio.get_writer(output_path, fps=fps, bitrate=bitrate, codec='libx264')
        
        i = 0
        for frame in reader:
            writer.append_data(frame)
            i += 1
            if i % 100 == 0:
                print(f"Processed {i} frames...")
                sys.stdout.flush()
        
        writer.close()
        reader.close()
        
        new_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Finished! New Size: {new_size:.2f} MB")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("You might need to install imageio-ffmpeg: pip install imageio-ffmpeg")

if __name__ == "__main__":
    input_file = "images/Conflict-Induced-Food-Crisis-Prediction01.mp4"
    output_file = "images/Conflict-Induced-Food-Crisis-Prediction01_compressed.mp4"
    # 600k bitrate is very safe for < 10MB if video < 120 seconds
    compress_video_imageio(input_file, output_file, bitrate="600k")
