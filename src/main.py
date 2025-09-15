"""
main.py
-------
Entry point to run predictions.
"""

import argparse
from src.predictor import predict_pose, process_video, realtime_pose_prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Head Pose Estimation")
    parser.add_argument("--mode", choices=["image", "video", "realtime"], required=True)
    parser.add_argument("--input", help="Path to image or video")
    parser.add_argument("--output", help="Output path (for video)")
    args = parser.parse_args()

    if args.mode == "image":
        predict_pose(args.input)
    elif args.mode == "video":
        process_video(args.input, args.output)
    elif args.mode == "realtime":
        realtime_pose_prediction()