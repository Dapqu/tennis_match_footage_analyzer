from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path, color=(0, 255, 0)):
        self.model = YOLO(model_path)
        self.color = color

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolates missing ball positions in a list of ball position dictionaries.
        
        Parameters:
        - ball_positions: List of dictionaries containing ball positions with frame numbers as keys.
        
        Returns:
        - ball_positions: List of dictionaries with interpolated ball positions.
        """

        ball_positions = [x.get(1, []) for x in ball_positions]
        # Convert the list above into panda's dataframe
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        # Interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        # Backfill the remaining missing values (if any) after interpolation
        df_ball_positions = df_ball_positions.bfill()

        # Convert the dataframe back to list
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detect balls in video frames. Optionally read from or write to a stub file.
        
        Parameters:
        - frames: List of video frames
        - read_from_stub: Boolean, if True, read detection results from a stub file
        - stub_path: Path to the stub file for reading/writing detection results
        
        Returns:
        - List of detected balls in each frame
        """
        
        # Initialize an empty list to store detected balls in each frame
        balls = []

        # Read detected balls from the stub file if specified
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                balls = pickle.load(f)
            return balls

        # Add detected ball in each frame to the balls list
        for frame in frames:
            ball = self.detect_frame(frame)
            balls.append(ball)

        # Write detected balls to the stub file if specified
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(balls, f)

        return balls

    def detect_frame(self, frame):
        """
        Detect balls in a single frame using the YOLO model.
        
        Parameters:
        - frame: A single video frame
        
        Returns:
        - Dictionary of detected balls with track IDs as keys and bounding box coordinates as values
        """

        # Use the YOLO model to track objects in the frame
        results = self.model.predict(frame, conf=0.15)[0]

        # Initialize an empty dictionary to store detected ball info, track_id as key and box_coords as content
        ball = {}

        # Loop through each detected box in the results
        for box in results.boxes:
            # Get the bounding box coordinates
            box_coords = box.xyxy.tolist()[0]
            # If the detected object is a person, add it to the ball dictionary
            ball[1] = box_coords

        return ball
    
    def draw_boxes(self, video_frames, balls):
        """
        Draw bounding boxes around detected balls in video frames.
        
        Parameters:
        - video_frames: List of video frames
        - balls: List of dictionaries containing detected ball info
        
        Returns:
        - List of video frames with bounding boxes drawn around detected balls
        """

        # Initialize an empty list to store annotated video frames
        output_video_frames = []

        # Loop through each frame and corresponding balls
        for frame, balls in zip(video_frames, balls):
            for track_id, bbox in balls.items():
                # Get the bounding box coordinates
                x1, y1, x2, y2 = bbox
                # Draw the ball ID above the bounding box
                cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_COMPLEX, 0.9, self.color, 2)
                # Draw the bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), self.color, 2)
            output_video_frames.append(frame)
        
        return output_video_frames