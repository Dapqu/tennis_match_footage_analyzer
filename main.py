from utils import (read_video, 
                   save_video)

from trackers import (PlayerTracker, 
                      BallTracker)

from court_line_detector import CourtLineDetector

def main():
    # Read video
    video_input_path = "assets/input_video.mp4"
    video_frames = read_video(video_input_path)

    # Detect players and ball
    ## Initialize PlayerTracker and BallTracker with draw color
    player_tracker = PlayerTracker(model_path='yolov8x', color=(0, 255, 0))
    ball_tracker = BallTracker(model_path='models/last.pt', color=(0, 0, 255))
    
    players = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")
    ball = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")

    # Court line detector
    court_model_path = "models/keypoints.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Draw output
    ## Draw players and ball bounding boxes
    output_video_frames = player_tracker.draw_boxes(video_frames, players)
    output_video_frames = ball_tracker.draw_boxes(output_video_frames, ball)
    ## Draw court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Save video
    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()