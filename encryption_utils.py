import os
import json
from cryptography.fernet import Fernet
from datetime import datetime
import uuid
import time


# Write metadata to a file
def write_metadata(metadata_path, metadata):
    with open(metadata_path, 'w') as file:
        json.dump(metadata, file)


# Encrypt a video
def encrypt_video(uploaded_file, video_name, video_description):
    try:
        start_time = time.time()
        video_data = uploaded_file.read()

        # Store original file extension
        original_extension = os.path.splitext(uploaded_file.name)[1]

        key = Fernet.generate_key()
        cipher = Fernet(key)
        encrypted_data = cipher.encrypt(video_data)

        encrypted_path = os.path.join("encrypted_videos", f"{video_name}.encrypted")
        key_path = os.path.join("metadata_key", f"{video_name}.key")

        with open(encrypted_path, 'wb') as file:
            file.write(encrypted_data)
        with open(key_path, 'wb') as file:
            file.write(key)

        metadata = {
            "video_ID": str(uuid.uuid4()),
            "Metadata": video_description,
            "original_extension": original_extension,
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        metadata_path = os.path.join("metadata_key", f"{video_name}.metadata")
        write_metadata(metadata_path, metadata)

        return encrypted_path, key_path, time.time() - start_time

    except Exception as e:
        print(f"Encryption error: {str(e)}")
        return None, None, 0


# Decrypt a video

def decrypt_video(encrypted_file, key_file, video_name):
    try:
        start_time = time.time()
        key = key_file.read()
        cipher = Fernet(key)

        encrypted_data = encrypted_file.read()
        decrypted_data = cipher.decrypt(encrypted_data)

        # Try to get original extension from metadata
        metadata_path = os.path.join("metadata_key", f"{video_name}.metadata")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as file:
                metadata = json.load(file)
                extension = metadata.get('original_extension', '.mp4')
        else:
            extension = '.mp4'  # Default to .mp4

        # Ensure extension starts with a dot
        if not extension.startswith('.'):
            extension = '.' + extension

        # Create a unique filename to avoid overwriting
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        decrypted_path = os.path.join("decrypted_videos", f"{video_name}_decrypted_{timestamp}{extension}")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(decrypted_path), exist_ok=True)

        # Write decrypted data to file
        with open(decrypted_path, 'wb') as file:
            file.write(decrypted_data)

        # Verify the file exists and has content
        if not os.path.exists(decrypted_path):
            print(f"Decryption failed: File not created at {decrypted_path}")
            return None, 0

        if os.path.getsize(decrypted_path) == 0:
            print(f"Decryption failed: File is empty at {decrypted_path}")
            return None, 0

        print(f"Successfully decrypted to {decrypted_path} ({os.path.getsize(decrypted_path)} bytes)")
        return decrypted_path, time.time() - start_time

    except Exception as e:
        print(f"Decryption error: {str(e)}")
        return None, 0