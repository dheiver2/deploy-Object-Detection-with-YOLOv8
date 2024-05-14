import numpy as np
import supervision as sv
from ultralytics import YOLO
import requests
from PIL import Image
import cv2
import os
import torch
import pandas as pd
import mss
from pathlib import Path
from io import BytesIO
import glob
import pafy
import csv

# Initialize YOLO model
model = YOLO("yolov8s.pt")

def process_frame(frame: np.ndarray, _) -> np.ndarray:
    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_ultralytics(results)
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    labels = []
    
    for i in range(len(detections.xyxy)):
        bbox = detections.xyxy[i]
        confidence = detections.confidence[i]
        class_id = detections.class_id[i]
        label = f"{model.names[class_id]} {confidence:0.2f}"
        labels.append(label)
    
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    return frame

def process_image(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return
    processed_image = process_frame(image, None)
    output_path = "result_image.jpg"
    cv2.imwrite(output_path, processed_image)
    print(f"Processed image saved to {output_path}")

def process_video(video_path: str):
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist")
        return
    sv.process_video(source_path=video_path, target_path="result.mp4", callback=process_frame)
    print("Processed video saved to result.mp4")

def process_http_image(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error: Unable to download image from {url}. {e}")
        return
    image = Image.open(BytesIO(response.content))
    image = np.array(image)
    processed_image = process_frame(image, None)
    output_path = "result_image.jpg"
    cv2.imwrite(output_path, processed_image)
    print(f"Processed image saved to {output_path}")

def process_screenshot():
    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[1])
        image = np.array(screenshot)
        processed_image = process_frame(image, None)
        output_path = "result_screenshot.jpg"
        cv2.imwrite(output_path, processed_image)
        print(f"Processed screenshot saved to {output_path}")

def process_pil_image(pil_image):
    if not isinstance(pil_image, Image.Image):
        print("Error: Provided input is not a PIL.Image object")
        return
    image = np.array(pil_image)
    processed_image = process_frame(image, None)
    output_path = "result_image.jpg"
    cv2.imwrite(output_path, processed_image)
    print(f"Processed image saved to {output_path}")

def process_opencv_image(cv_image):
    if not isinstance(cv_image, np.ndarray):
        print("Error: Provided input is not an OpenCV image (numpy array)")
        return
    processed_image = process_frame(cv_image, None)
    output_path = "result_image.jpg"
    cv2.imwrite(output_path, processed_image)
    print(f"Processed image saved to {output_path}")

def process_numpy_array(np_array):
    if not isinstance(np_array, np.ndarray):
        print("Error: Provided input is not a numpy array")
        return
    processed_image = process_frame(np_array, None)
    output_path = "result_image.jpg"
    cv2.imwrite(output_path, processed_image)
    print(f"Processed image saved to {output_path}")

def process_torch_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        print("Error: Provided input is not a torch tensor")
        return
    # Convert BCHW format with RGB channels float32 (0.0-1.0) to HWC format with BGR uint8 (0-255)
    np_array = tensor[0].permute(1, 2, 0).mul(255).byte().numpy()
    np_array = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)
    process_numpy_array(np_array)

def process_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} does not exist")
        return
    df = pd.read_csv(csv_path)
    for path in df['path']:
        main(path)

def process_directory(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Error: Directory {directory_path} does not exist")
        return
    for file_path in Path(directory_path).rglob('*'):
        main(str(file_path))

def process_glob_pattern(pattern):
    for file_path in glob.glob(pattern):
        main(file_path)

def process_youtube(url):
    try:
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")
        cap = cv2.VideoCapture(best.url)
    except Exception as e:
        print(f"Error: Unable to process YouTube video. {e}")
        return

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        processed_frame = process_frame(frame, None)
        cv2.imshow('YouTube Stream', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def process_live_stream(stream_url: str):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Error: Could not open video stream from {stream_url}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        processed_frame = process_frame(frame, None)
        cv2.imshow('Live Stream', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def process_multi_stream(stream_list_path: str):
    if not os.path.exists(stream_list_path):
        print(f"Error: Stream list file {stream_list_path} does not exist")
        return

    with open(stream_list_path, 'r') as file:
        streams = file.readlines()
    
    caps = [cv2.VideoCapture(stream.strip()) for stream in streams]
    
    while True:
        for cap in caps:
            if not cap.isOpened():
                print("Error: Could not open video stream.")
                continue
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                continue
            
            processed_frame = process_frame(frame, None)
            cv2.imshow(f'Stream {caps.index(cap)}', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

# Main function to determine the type of input
def main(input_path):
    if isinstance(input_path, str):
        if input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            process_image(input_path)
        elif input_path.lower().endswith('.mp4'):
            process_video(input_path)
        elif input_path.startswith('http://') or input_path.startswith('https://'):
            if 'youtube' in input_path:
                process_youtube(input_path)
            elif 'jpg' in input_path or 'jpeg' in input_path or 'png' in input_path:
                process_http_image(input_path)
            else:
                process_live_stream(input_path)
        elif input_path.lower() == 'screen':
            process_screenshot()
        elif input_path.lower().endswith('.csv'):
            process_csv(input_path)
        elif Path(input_path).is_dir():
            process_directory(input_path)
        elif '*' in input_path:
            process_glob_pattern(input_path)
        elif input_path.lower().endswith('.streams'):
            process_multi_stream(input_path)
        elif input_path == 'webcam':
            process_live_stream(0)
        else:
            print("Unsupported input format. Please provide a valid image, video file, HTTP URL, or 'webcam' for live stream from webcam.")
    elif isinstance(input_path, Image.Image):
        process_pil_image(input_path)
    elif isinstance(input_path, np.ndarray):
        process_opencv_image(input_path)
    elif isinstance(input_path, torch.Tensor):
        process_torch_tensor(input_path)
    else:
        print("Unsupported input type.")

# Example usage:
# main("/kaggle/input/uavid-v1/uavid_test/seq42/Images/000000.png")
# main("/path/to/video.mp4")
# main("http://example.com/path/to/image.jpg")
# main("http://example.com/path/to/live/stream")
# main("screen")
# main("path/to/csv_file.csv")
# main("path/to/directory")
main("/kaggle/input/uavid-v1/uavid_val/seq16/Images/000000.png")
# main("https://youtu.be/LNwODJXcvt4")
# main("path/to/list.streams")
# main("webcam")

