import os
import subprocess
import shutil

def get_video_framerate(video_path: str) -> float:
    """Gets the framerate of a video using ffprobe."""
    if not shutil.which("ffprobe"):
        print("Error: ffprobe not found. Please install ffmpeg.")
        return None
    
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error getting framerate: {result.stderr}")
        return None
    
    try:
        num, den = map(int, result.stdout.strip().split('/'))
        return num / den
    except (ValueError, ZeroDivisionError) as e:
        print(f"Could not parse framerate from ffprobe output: '{result.stdout.strip()}' due to {e}")
        return None

def slice_video(video_path: str, start_time: float, end_time: float, output_path: str):
    """
    Slices a video file from start_time to end_time using ffmpeg and saves it to output_path.
    This version uses libx264 for better compatibility.

    Args:
        video_path (str): Path to the input video file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        output_path (str): Path to save the sliced video.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found. Please install ffmpeg.")
        return
        
    fps = get_video_framerate(video_path)
    if fps is None:
        return
        
    duration = end_time - start_time
    total_frames_in_slice = int(duration * fps)

    print(f"Original framerate: {fps:.2f} FPS")
    print(f"Total frames in new slice: {total_frames_in_slice}")
    
    if os.path.exists(output_path):
        os.remove(output_path)

    command = [
        "ffmpeg",
        "-i", video_path,
        "-ss", str(start_time),
        "-to", str(end_time),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-crf", "0",
        output_path,
    ]
    
    print(f"\nRunning ffmpeg command:\n{' '.join(command)}\n")
    
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error slicing video with ffmpeg:")
        print(result.stderr)
    else:
        print(f"Sliced video saved to {output_path}") 