from ultralytics import YOLO
import cv2
import pickle

class PlayerTracker:
    def __init__(self, model_path, color=(0, 255, 0)):
        self.model = YOLO(model_path)
        self.color = color

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detect players in video frames. Optionally read from or write to a stub file.
        
        Parameters:
        - frames: List of video frames
        - read_from_stub: Boolean, if True, read detection results from a stub file
        - stub_path: Path to the stub file for reading/writing detection results
        
        Returns:
        - List of detected players in each frame
        """
        
        # Initialize an empty list to store detected players in each frame
        players = []

        # Read detected players from the stub file if specified
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                players = pickle.load(f)
            return players

        # Add detected player in each frame to the players list
        for frame in frames:
            player = self.detect_frame(frame)
            players.append(player)

        # Write detected players to the stub file if specified
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(players, f)

        return players

    def detect_frame(self, frame):
        """
        Detect players in a single frame using the YOLO model.
        
        Parameters:
        - frame: A single video frame
        
        Returns:
        - Dictionary of detected players with track IDs as keys and bounding box coordinates as values
        """

        # Use the YOLO model to track objects in the frame
        results = self.model.track(frame, persist=True)[0]
        # Get the id names of the detected classes
        id_names = results.names

        # Initialize an empty dictionary to store detected player info, track_id as key and box_coords as content
        player = {}

        # Loop through each detected box in the results
        for box in results.boxes:
            # Get the tracking ID for the detected object
            track_id = int(box.id.tolist()[0])
            # Get the bounding box coordinates
            box_coords = box.xyxy.tolist()[0]
            # Get the class ID of the detected object
            object_cls_id = box.cls.tolist()[0]
            # Get the class name of the detected object
            object_cls_name = id_names[object_cls_id]

            # If the detected object is a person, add it to the player dictionary
            if object_cls_name == "person":
                player[track_id] = box_coords

        return player
    
    def draw_boxes(self, video_frames, players):
        """
        Draw bounding boxes around detected players in video frames.
        
        Parameters:
        - video_frames: List of video frames
        - players: List of dictionaries containing detected player info
        
        Returns:
        - List of video frames with bounding boxes drawn around detected players
        """

        # Initialize an empty list to store annotated video frames
        output_video_frames = []

        # Loop through each frame and corresponding players
        for frame, players in zip(video_frames, players):
            for track_id, bbox in players.items():
                # Get the bounding box coordinates
                x1, y1, x2, y2 = bbox
                # Draw the player ID above the bounding box
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_COMPLEX, 0.9, self.color, 2)
                # Draw the bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), self.color, 2)
            output_video_frames.append(frame)
        
        return output_video_frames