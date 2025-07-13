import os
from ultralytics import YOLO
import cv2


# In detection.py:
def load_model():
    # Try multiple potential model locations
    potential_paths = [
        os.path.join('models', 'best.pt'),  # Local models directory
        os.path.join(os.path.expanduser('~'), 'Downloads', 'AnomalyDetection', 'runs', 'best.pt'),  # Original path
        'yolov8n.pt'  # Fallback to default model
    ]

    for path in potential_paths:
        if os.path.exists(path):
            return YOLO(path)

    # If no model is found, use YOLO's default model
    print("No local model found, using YOLO default model")
    return YOLO("yolov8n.pt")


def process_video(video_path, output_path):
    # Load model only when needed
    model = load_model()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    threshold = 0.5

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_file = os.path.join(output_path, f'{os.path.basename(video_path)}_out.mp4')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (W, H))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                label = f'{results.names[int(class_id)].upper()} {score:.2f}'
                cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def process_folder(test_folder, result_folder):
    """Function to process all videos in a folder"""
    os.makedirs(result_folder, exist_ok=True)

    for file_name in os.listdir(test_folder):
        if file_name.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            video_path = os.path.join(test_folder, file_name)
            process_video(video_path, result_folder)


# Only run this if the script is run directly (not imported)
if __name__ == "__main__":
    test_folder = os.path.join('/', 'home', 'abdullah', 'Downloads', 'new', 'data', 'test_videos')
    result_folder = os.path.join('/', 'home', 'abdullah', 'Downloads', 'new', 'data', 'test_videos', 'result')
    process_folder(test_folder, result_folder)