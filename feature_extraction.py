import hashlib
import os
import json
import cv2
import numpy as np
from cryptography.fernet import Fernet
from detection import load_model


class FeatureExtractor:
    def __init__(self):
        """Initialize the feature extractor with optional YOLO model"""
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)

        # Load YOLO model if provided
        model = load_model()
        if model:
            self.model = model
        else:
            self.model = None
            raise "Model best.pt not found for feature extraction"

    def encrypt(self, data):
        """Encrypt data using Fernet"""
        return self.cipher.encrypt(data.encode())

    def decrypt(self, data):
        """Decrypt data using Fernet"""
        return self.cipher.decrypt(data).decode()

    def hash_feature(self, feature):
        """Hash a feature using SHA-256"""
        return hashlib.sha256(str(feature).encode()).hexdigest()

    def extract_features(self, video_path, save_frames=False, custom_video_name=None):
        """Extract features from video frames using YOLO model"""
        if not self.model:
            model = load_model()
            if model:
                self.model = model
            else:
                # Fall back to a simpler approach if model doesn't exist
                raise "Model best.pt not found for feature extraction"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frame_features = []
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Use custom_video_name if provided, otherwise use the original file name
        video_name = custom_video_name if custom_video_name else os.path.splitext(os.path.basename(video_path))[0]

        # Create directory for detected frames if saving frames
        frames_dir = None
        if save_frames:
            frames_dir = os.path.join("detected_frames", video_name)
            os.makedirs(frames_dir, exist_ok=True)
            print(f"Created frames directory: {frames_dir}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every 5th frame to reduce computation
            if frame_count % 5 == 0:
                # Get timestamp
                seconds = frame_count / fps
                timestamp = f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"

                # Extract features using YOLO
                results = self.model(frame)[0]
                features = []
                has_anomaly = False

                # Extract class names and confidences
                for result in results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = result
                    if score > 0.5:  # Confidence threshold
                        class_name = results.names[int(class_id)].upper()
                        features.append(class_name)
                        has_anomaly = True

                        # Draw bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f'{class_name} {score:.2f}'
                        cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                # Save frame if it has anomalies and save_frames is enabled
                if has_anomaly and save_frames and frames_dir:
                    frame_path = os.path.join(frames_dir, f"frame_{frame_count}_{timestamp.replace(':', '-')}.jpg")
                    print(f"Saving frame to: {frame_path}")
                    cv2.imwrite(frame_path, frame)

                    # Add frame path to features
                    frame_features.append({
                        'frame_id': frame_count,
                        'timestamp': timestamp,
                        'features': features,
                        'frame_path': frame_path
                    })
                elif has_anomaly:
                    frame_features.append({
                        'frame_id': frame_count,
                        'timestamp': timestamp,
                        'features': features
                    })

            frame_count += 1

        cap.release()
        print(f"Extracted {len(frame_features)} feature frames from {video_path}")
        return frame_features

    def create_secure_index(self, frame_features):
        """Create a secure index of video frames based on hashed features"""
        secure_index = {}
        for frame in frame_features:
            hashed_features = set(self.hash_feature(f) for f in frame['features'])
            encrypted_frame_id = self.encrypt(str(frame['frame_id']))
            frame_data = {
                'hashed_features': list(hashed_features),
                'timestamp': frame['timestamp']
            }

            # Add frame path if available
            if 'frame_path' in frame:
                frame_data['frame_path'] = frame['frame_path']

            secure_index[encrypted_frame_id.decode()] = frame_data
        return secure_index

    def save_secure_index(self, secure_index, video_name):
        """Save the secure index to a file"""
        os.makedirs("feature_indices", exist_ok=True)
        index_path = os.path.join("feature_indices", f"{video_name}_index.json")

        # Save the key
        key_path = os.path.join("feature_indices", f"{video_name}_feature_key.key")
        with open(key_path, 'wb') as key_file:
            key_file.write(self.key)

        # Save the index
        with open(index_path, 'w') as index_file:
            json.dump(secure_index, index_file)

        return index_path, key_path

    def load_secure_index(self, video_name, key_path=None):
        """Load a secure index from a file"""
        index_path = os.path.join("feature_indices", f"{video_name}_index.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")

        # Load the key if provided
        if key_path and os.path.exists(key_path):
            with open(key_path, 'rb') as key_file:
                self.key = key_file.read()
                self.cipher = Fernet(self.key)

        # Load the index
        with open(index_path, 'r') as index_file:
            secure_index = json.load(index_file)

        return secure_index

    def query_features(self, secure_index, query_features):
        """Perform secure context-based query retrieval"""
        hashed_query = set(self.hash_feature(f) for f in query_features)
        retrieved_frames = []

        for encrypted_frame_id, data in secure_index.items():
            hashed_features = set(data['hashed_features'])
            if len(hashed_query.intersection(hashed_features)) / len(
                    hashed_query) >= 0.5:  # Match if at least 50% of query features match
                try:
                    decrypted_frame_id = self.decrypt(encrypted_frame_id.encode())
                    frame_info = {
                        'frame_id': decrypted_frame_id,
                        'timestamp': data['timestamp']
                    }

                    # Add frame path if available
                    if 'frame_path' in data:
                        frame_info['frame_path'] = data['frame_path']

                    retrieved_frames.append(frame_info)
                except Exception as e:
                    print(f"Error decrypting frame ID: {str(e)}")

        return retrieved_frames


def process_video_features(video_path, video_name, video_name_for_dir=None):
    """Process a video and extract features"""
    extractor = FeatureExtractor()
    try:
        # Extract features and save detected frames
        frame_features = extractor.extract_features(
            video_path,
            save_frames=True,
            custom_video_name=video_name_for_dir or video_name
        )

        # Create secure index
        secure_index = extractor.create_secure_index(frame_features)

        # Save secure index
        index_path, key_path = extractor.save_secure_index(secure_index, video_name)

        return {
            'status': 'success',
            'message': f"Features extracted and indexed for {video_path}",
            'index_path': index_path,
            'key_path': key_path,
            'feature_count': len(frame_features)
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Error extracting features: {str(e)}"
        }


def search_video_features(video_name, query_features, key_path=None):
    """Search for features in a video's secure index"""
    extractor = FeatureExtractor()
    try:
        # Load secure index
        secure_index = extractor.load_secure_index(video_name, key_path)

        # Query features
        retrieved_frames = extractor.query_features(secure_index, query_features)

        return {
            'status': 'success',
            'message': f"Found {len(retrieved_frames)} matching frames",
            'frames': retrieved_frames
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Error searching features: {str(e)}"
        }


def time_to_seconds(time_str):
    """Convert a timestamp string (HH:MM:SS) to seconds"""
    try:
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except ValueError:
        # Handle potential formatting errors
        return 0


def seconds_to_time(seconds):
    """Convert seconds to a timestamp string (HH:MM:SS)"""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_frames_by_timerange(video_name, start_time, end_time):
    """
    Get detected frames within a specific time range for a video

    Args:
        video_name: Name of the video
        start_time: Start timestamp (HH:MM:SS)
        end_time: End timestamp (HH:MM:SS)

    Returns:
        Dictionary containing frame information filtered by time range
    """
    # Get all frames first
    frames_result = get_detected_frames(video_name)

    if frames_result['status'] != 'success' or not frames_result['frames']:
        return frames_result

    # Convert time strings to seconds for comparison
    start_seconds = time_to_seconds(start_time)
    end_seconds = time_to_seconds(end_time)

    # Filter frames by timestamp
    filtered_frames = []
    for frame in frames_result['frames']:
        # Convert frame timestamp to seconds
        frame_time = frame['timestamp']
        frame_seconds = time_to_seconds(frame_time)

        # Check if frame is within range
        if start_seconds <= frame_seconds <= end_seconds:
            filtered_frames.append(frame)

    return {
        'status': 'success' if filtered_frames else 'error',
        'message': f"Found {len(filtered_frames)} frames between {start_time} and {end_time}" if filtered_frames else f"No frames found between {start_time} and {end_time}",
        'frames': filtered_frames
    }


def get_detected_frames(video_name):
    """Get all detected frames for a video"""
    try:
        # First, try with the original video_name
        frames_dir = os.path.join("detected_frames", video_name)

        # If that directory doesn't exist or is empty, try with potential alternatives
        if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
            # Check for potential alternative directories
            alternative_dirs = []

            # Check if there's a directory with _out suffix
            out_dir = os.path.join("detected_frames", f"{video_name}_out")
            if os.path.exists(out_dir):
                alternative_dirs.append(out_dir)

            # Check if there's a directory that ends with _out.mp4
            mp4_out_dir = os.path.join("detected_frames", f"{video_name}_out.mp4")
            if os.path.exists(mp4_out_dir):
                alternative_dirs.append(mp4_out_dir)

            # Use the first valid alternative that has frames
            for alt_dir in alternative_dirs:
                if os.path.exists(alt_dir) and len(os.listdir(alt_dir)) > 0:
                    frames_dir = alt_dir
                    break

        # If directory still doesn't exist, create it
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir, exist_ok=True)
            return {
                'status': 'error',
                'message': f"No detected frames found for {video_name}",
                'frames': []
            }

        frames = []
        for frame_file in os.listdir(frames_dir):
            if frame_file.endswith('.jpg'):
                # Extract frame info from filename
                # Format: frame_<frame_id>_<timestamp>.jpg
                parts = frame_file.split('_')
                if len(parts) >= 3:
                    frame_id = parts[1]
                    timestamp = parts[2].replace('.jpg', '').replace('-', ':')

                    frames.append({
                        'frame_id': frame_id,
                        'timestamp': timestamp,
                        'frame_path': os.path.join(frames_dir, frame_file)
                    })

        return {
            'status': 'success' if frames else 'error',
            'message': f"Found {len(frames)} detected frames" if frames else f"No detected frames found for {video_name}",
            'frames': frames
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Error retrieving detected frames: {str(e)}",
            'frames': []
        }


def get_anonymized_detected_frames(video_name, anonymize=True):
    """Get all detected frames for a video with option to anonymize

    Args:
        video_name: Name of the video
        anonymize: Whether to return anonymized frames (blurred)

    Returns:
        Dictionary containing frame information
    """
    from anonymization import anonymize_image_file

    try:
        # First, try with the original video_name
        frames_dir = os.path.join("detected_frames", video_name)

        # If that directory doesn't exist or is empty, try with potential alternatives
        if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
            # Check for potential alternative directories
            alternative_dirs = []

            # Check if there's a directory with _out suffix
            out_dir = os.path.join("detected_frames", f"{video_name}_out")
            if os.path.exists(out_dir):
                alternative_dirs.append(out_dir)

            # Check if there's a directory that ends with _out.mp4
            mp4_out_dir = os.path.join("detected_frames", f"{video_name}_out.mp4")
            if os.path.exists(mp4_out_dir):
                alternative_dirs.append(mp4_out_dir)

            # Use the first valid alternative that has frames
            for alt_dir in alternative_dirs:
                if os.path.exists(alt_dir) and len(os.listdir(alt_dir)) > 0:
                    frames_dir = alt_dir
                    break

        # If directory still doesn't exist, create it
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir, exist_ok=True)
            return {
                'status': 'error',
                'message': f"No detected frames found for {video_name}",
                'frames': []
            }

        # Create directory for anonymized frames if needed
        anonymized_frames_dir = None
        if anonymize:
            anonymized_frames_dir = os.path.join("anonymized_frames", video_name)
            os.makedirs(anonymized_frames_dir, exist_ok=True)

        frames = []
        for frame_file in os.listdir(frames_dir):
            if frame_file.endswith('.jpg'):
                # Extract frame info from filename
                # Format: frame_<frame_id>_<timestamp>.jpg
                parts = frame_file.split('_')
                if len(parts) >= 3:
                    frame_id = parts[1]
                    timestamp = parts[2].replace('.jpg', '').replace('-', ':')

                    original_frame_path = os.path.join(frames_dir, frame_file)

                    # If anonymization is requested, create or use anonymized frame
                    if anonymize:
                        anonymized_frame_path = os.path.join(anonymized_frames_dir, frame_file)

                        # Check if anonymized frame already exists, if not create it
                        if not os.path.exists(anonymized_frame_path):
                            anonymized_frame_path = anonymize_image_file(
                                original_frame_path,
                                anonymized_frame_path
                            )

                        # Use anonymized path if successful, otherwise fall back to original
                        frame_path = anonymized_frame_path if anonymized_frame_path else original_frame_path
                    else:
                        frame_path = original_frame_path

                    frames.append({
                        'frame_id': frame_id,
                        'timestamp': timestamp,
                        'frame_path': frame_path,
                        'is_anonymized': anonymize
                    })

        return {
            'status': 'success' if frames else 'error',
            'message': f"Found {len(frames)} detected frames" if frames else f"No detected frames found for {video_name}",
            'frames': frames
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Error retrieving detected frames: {str(e)}",
            'frames': []
        }

# In feature_extraction.py - Add a new function for anonymized timestamp search
def get_anonymized_frames_by_timerange(video_name, start_time, end_time, anonymize=True):
    """
    Get detected frames within a specific time range for a video with option to anonymize

    Args:
        video_name: Name of the video
        start_time: Start timestamp (HH:MM:SS)
        end_time: End timestamp (HH:MM:SS)
        anonymize: Whether to return anonymized frames (blurred)

    Returns:
        Dictionary containing frame information filtered by time range
    """
    # Get all frames first with anonymization option
    from anonymization import anonymize_image_file

    # Convert time strings to seconds for comparison
    start_seconds = time_to_seconds(start_time)
    end_seconds = time_to_seconds(end_time)

    try:
        # First, try with the original video_name
        frames_dir = os.path.join("detected_frames", video_name)

        # If that directory doesn't exist or is empty, try with potential alternatives
        if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
            # Check for potential alternative directories
            alternative_dirs = []

            # Check if there's a directory with _out suffix
            out_dir = os.path.join("detected_frames", f"{video_name}_out")
            if os.path.exists(out_dir):
                alternative_dirs.append(out_dir)

            # Check if there's a directory that ends with _out.mp4
            mp4_out_dir = os.path.join("detected_frames", f"{video_name}_out.mp4")
            if os.path.exists(mp4_out_dir):
                alternative_dirs.append(mp4_out_dir)

            # Use the first valid alternative that has frames
            for alt_dir in alternative_dirs:
                if os.path.exists(alt_dir) and len(os.listdir(alt_dir)) > 0:
                    frames_dir = alt_dir
                    break

        # If directory still doesn't exist, create it
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir, exist_ok=True)
            return {
                'status': 'error',
                'message': f"No detected frames found for {video_name}",
                'frames': []
            }

        # Create directory for anonymized frames if needed
        anonymized_frames_dir = None
        if anonymize:
            anonymized_frames_dir = os.path.join("anonymized_frames", video_name)
            os.makedirs(anonymized_frames_dir, exist_ok=True)

        frames = []
        for frame_file in os.listdir(frames_dir):
            if frame_file.endswith('.jpg'):
                # Extract frame info from filename
                # Format: frame_<frame_id>_<timestamp>.jpg
                parts = frame_file.split('_')
                if len(parts) >= 3:
                    frame_id = parts[1]
                    timestamp = parts[2].replace('.jpg', '').replace('-', ':')

                    # Convert frame timestamp to seconds for filtering
                    frame_seconds = time_to_seconds(timestamp)

                    # Only process frames within the requested time range
                    if start_seconds <= frame_seconds <= end_seconds:
                        original_frame_path = os.path.join(frames_dir, frame_file)

                        # If anonymization is requested, create or use anonymized frame
                        if anonymize:
                            anonymized_frame_path = os.path.join(anonymized_frames_dir, frame_file)

                            # Check if anonymized frame already exists, if not create it
                            if not os.path.exists(anonymized_frame_path):
                                anonymized_frame_path = anonymize_image_file(
                                    original_frame_path,
                                    anonymized_frame_path
                                )

                            # Use anonymized path if successful, otherwise fall back to original
                            frame_path = anonymized_frame_path if anonymized_frame_path else original_frame_path
                        else:
                            frame_path = original_frame_path

                        frames.append({
                            'frame_id': frame_id,
                            'timestamp': timestamp,
                            'frame_path': frame_path,
                            'is_anonymized': anonymize
                        })

        return {
            'status': 'success' if frames else 'error',
            'message': f"Found {len(frames)} frames between {start_time} and {end_time}" if frames else f"No frames found between {start_time} and {end_time}",
            'frames': frames
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Error retrieving detected frames: {str(e)}",
            'frames': []
        }
