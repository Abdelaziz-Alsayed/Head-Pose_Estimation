import cv2
import numpy as np
from .pose_utils import draw_pose
from .model import load_model


class HeadPosePredictor:
    """
    Head Pose Predictor

    Provides methods to predict head pose on:
    - Single image
    - Video file
    - Real-time webcam stream
    """

    def __init__(self, model_path: str):
        """
        Initialize the predictor with a trained model.

        Args:
            model_path (str): Path to the saved trained model
        """
        self.model = load_model(model_path)

    def predict_image(self, image_path: str, output_path: str = None):
        """
        Predict head pose on a single image.

        Args:
            image_path (str): Path to the input image
            output_path (str, optional): Save output with pose overlay
        """
        image = cv2.imread(image_path)
        angles = self.model.predict(image)   # Replace with your pipeline
        vis = draw_pose(image, angles)

        if output_path:
            cv2.imwrite(output_path, vis)
        return angles, vis

    def predict_video(self, video_path: str, output_path: str = "output_video.avi"):
        """
        Predict head pose on a video file.

        Args:
            video_path (str): Path to input video
            output_path (str): Path to save processed video
        """
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, 
                              (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            angles = self.model.predict(frame)  # Replace with your pipeline
            vis = draw_pose(frame, angles)
            out.write(vis)

        cap.release()
        out.release()

    def predict_realtime(self, cam_index: int = 0):
        """
        Predict head pose from a live webcam stream.

        Args:
            cam_index (int): Webcam index (default=0)
        """
        cap = cv2.VideoCapture(cam_index)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            angles = self.model.predict(frame)  # Replace with your pipeline
            vis = draw_pose(frame, angles)
            cv2.imshow("Head Pose Estimation", vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()