import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        """
        Initializes the CourtLineDetector with a pre-trained ResNet50 model.
        
        Parameters:
        - model_path: Path to the model weights file
        """

        # Load a pre-trained ResNet50 model and modify the final layer for 28 output values (14 keypoints)
        self.model = models.resnet50()
        # 14 keypoints with (x, y) coordinates
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) 
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        # Define the image transformation pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),        # Convert image to PIL format
            transforms.Resize((224, 224)),  # Resize image to 224x224
            transforms.ToTensor(),          # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize the image
        ])

    def predict(self, image):
        """
        Predicts keypoints on a single image.
        
        Parameters:
        - image: Input image (numpy array)
        
        Returns:
        - keypoints: Predicted keypoints (numpy array)
        """

        # Convert the image to RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Apply the transformation pipeline and add a batch dimension
        image_tensor = self.transform(image_rgb).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Extract keypoints and rescale them to the original image dimensions
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        return keypoints

    def draw_keypoints(self, image, keypoints):
        """
        Draws keypoints on a single image.
        
        Parameters:
        - image: Input image (numpy array)
        - keypoints: Predicted keypoints (numpy array)
        
        Returns:
        - image: Image with keypoints drawn
        """

        # Plot keypoints on the image
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            # Draw the keypoint index
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # Draw the keypoint as a circle
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        """
        Draws keypoints on each frame of a video.
        
        Parameters:
        - video_frames: List of video frames
        - keypoints: Predicted keypoints (numpy array)
        
        Returns:
        - output_video_frames: List of video frames with keypoints drawn
        """
        
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames