from utils import (read_video, 
                   save_video)

from trackers import PlayerTracker

def main():
    # Read video
    video_input_path = "assets/input_video.mp4"
    video_frames = read_video(video_input_path)

    # Detect players
    ## Initialize PlayerTracker with draw color
    player_tracker = PlayerTracker(model_path='yolov8x', color=(0, 255, 0))
    players = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")

    # Draw output
    ## Draw player bounding boxes
    output_video_frames = player_tracker.draw_boxes(video_frames, players)

    # Save video
    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()