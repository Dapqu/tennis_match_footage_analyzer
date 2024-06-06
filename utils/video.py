# OpenCV library
import cv2

def read_video(video_input_path):
    """
    Reads a video file and returns a list of frames.
    
    Parameters:
    - video_input_path: Path to the input video file
    
    Returns:
    - frames: List of video frames
    """

    vc = cv2.VideoCapture(video_input_path)
    frames = []

    while True:
        # Read a frame from the video
        ret, frame = vc.read()
        # If the frame could not be read (end of video)
        if not ret:
            break
        # Add the frame to list
        frames.append(frame)
    
    # Free the VideoCapture
    vc.release()
    return frames

def save_video(video_output_frames, video_output_path):
    """
    Saves a list of frames to a video file.
    
    Parameters:
    - video_output_frames: List of frames to be saved
    - video_output_path: Path to the output video file
    """

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vw = cv2.VideoWriter(video_output_path, fourcc, 24, (video_output_frames[0].shape[1], video_output_frames[0].shape[0]))
    
    # Write each frame to the output video
    for frame in video_output_frames:
        vw.write(frame)
    
    # Free the VideoWrite
    vw.release()