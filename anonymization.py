import cv2
import time
import os
import numpy as np


def anonymize_video(video_path):
    try:
        start_time = time.time()
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return None, 0

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a directory for anonymized videos if it doesn't exist
        anonymized_dir = "anonymized_videos"
        os.makedirs(anonymized_dir, exist_ok=True)

        # Define output path
        filename = os.path.basename(video_path)
        base_name = os.path.splitext(filename)[0]

        # Try multiple codecs in order of preference
        codecs_to_try = [
            # Codec, extension, fourcc
            ('mp4v', 'mp4', cv2.VideoWriter_fourcc(*'mp4v')),  # More widely supported codec
            ('XVID', 'avi', cv2.VideoWriter_fourcc(*'XVID')),  # Common AVI codec
            ('MJPG', 'avi', cv2.VideoWriter_fourcc(*'MJPG')),  # Motion JPEG
            ('DIV3', 'avi', cv2.VideoWriter_fourcc(*'DIV3'))  # DivX3
        ]

        desired_size = (width, height)  # Maintain original dimensions
        out = None
        output_path = None

        # Try each codec until one works
        for codec_name, ext, fourcc in codecs_to_try:
            output_path = os.path.join(anonymized_dir, f"{base_name}_anonymized.{ext}")
            print(f"Trying codec {codec_name} with output file {output_path}")

            out = cv2.VideoWriter(output_path, fourcc, fps, desired_size)
            if out.isOpened():
                print(f"Successfully opened video writer with {codec_name} codec")
                break
            else:
                print(f"Failed to create output video with {codec_name} codec, trying next...")

        if not out.isOpened():
            print("Failed to open video writer with any codec. Check OpenCV installation and codecs.")
            return None, 0

        # Define pixelation parameters
        pixelation_block_size = (32, 18)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply pixelation effect
            output_frame = anonymize_frame(frame, pixelation_block_size)

            # Write the frame to output video
            out.write(output_frame)
            frame_count += 1

        # Release resources
        cap.release()
        out.release()

        processing_time = time.time() - start_time
        print(f"Anonymization completed: {frame_count} frames processed in {processing_time:.2f} seconds")
        print(f"Saved anonymized video to: {output_path}")

        if os.path.exists(output_path):
            print(f"File exists with size: {os.path.getsize(output_path)} bytes")
            return output_path, processing_time
        else:
            print(f"File was not created at {output_path}")
            return None, processing_time

    except Exception as e:
        print(f'Anonymization Error: {str(e)}')
        return None, 0


def anonymize_frame(frame, block_size=(32, 18)):
    """Anonymize a single frame by pixelation

    Args:
        frame: The input frame to anonymize
        block_size: Tuple of (width, height) for pixelation blocks

    Returns:
        The anonymized frame
    """
    h, w = frame.shape[:2]
    # Apply pixelation effect
    pixelated = cv2.resize(frame, block_size, interpolation=cv2.INTER_NEAREST)
    anonymized = cv2.resize(pixelated, (w, h), interpolation=cv2.INTER_NEAREST)
    return anonymized


def anonymize_image_file(image_path, output_path=None, block_size=(32, 18)):
    """Anonymize an image file and save the result

    Args:
        image_path: Path to the input image file
        output_path: Path to save the anonymized image (if None, adds '_anonymized' suffix)
        block_size: Tuple of (width, height) for pixelation blocks

    Returns:
        Path to the anonymized image or None on failure
    """
    try:
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error reading image file: {image_path}")
            return None

        # Apply anonymization
        anonymized = anonymize_frame(frame, block_size)

        # Create output path if not provided
        if output_path is None:
            base_dir = os.path.dirname(image_path)
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(base_dir, f"{name}_anonymized{ext}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Save the anonymized image
        cv2.imwrite(output_path, anonymized)

        return output_path
    except Exception as e:
        print(f"Error anonymizing image: {str(e)}")
        return None