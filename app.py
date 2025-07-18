import os
import json
import streamlit as st
from io import BytesIO
from encryption_utils import encrypt_video, decrypt_video
from anonymization import anonymize_video
from search_utils import search_encrypted_videos
from detection import process_video
from feature_extraction import process_video_features, search_video_features, get_detected_frames, \
    get_anonymized_detected_frames

# Create necessary folders
os.makedirs("encrypted_videos", exist_ok=True)
os.makedirs("metadata_key", exist_ok=True)
os.makedirs("decrypted_videos", exist_ok=True)
os.makedirs("processed_videos", exist_ok=True)
os.makedirs("feature_indices", exist_ok=True)
os.makedirs("detected_frames", exist_ok=True)
os.makedirs("anonymized_frames", exist_ok=True)
os.makedirs("anonymized_videos", exist_ok=True)  # Ensure this directory exists


# Function to get all encrypted videos
def get_all_encrypted_videos():
    videos = []
    if os.path.exists("encrypted_videos"):
        for filename in os.listdir("encrypted_videos"):
            if filename.endswith(".encrypted"):
                videos.append(os.path.splitext(filename)[0])
    return videos


# Function to reset session state
def reset_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.current_step = 1
    st.session_state.encrypted_success = False
    st.session_state.decryption_key_uploaded = False
    st.session_state.decryption_success = False
    st.session_state.show_search = False
    st.session_state.encryption_key_generated = False
    st.session_state.encryption_key = None


def main():
    st.set_page_config(page_title="Privacy-Protected Anomaly Detection",
                       layout="wide")

    st.markdown("""
        <style>
        /* Make all Streamlit widget text bold */
        * {
            font-weight: bold !important;
        }

        /* Optional: Target Streamlit widgets more precisely */
        .stMarkdown, .stText, .stExpander, .stExpanderHeader, .stTextInput label, .stSelectbox label {
            font-weight: bold !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Privacy-Protected Anomaly Detection")
    # Initialize session state variables if they don't exist
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'encrypted_path' not in st.session_state:
        st.session_state.encrypted_path = None
    if 'key_path' not in st.session_state:
        st.session_state.key_path = None
    if 'encrypted_success' not in st.session_state:
        st.session_state.encrypted_success = False
    if 'decryption_key_uploaded' not in st.session_state:
        st.session_state.decryption_key_uploaded = False
    if 'decryption_success' not in st.session_state:
        st.session_state.decryption_success = False
    if 'show_search' not in st.session_state:
        st.session_state.show_search = False
    if 'encryption_key_generated' not in st.session_state:
        st.session_state.encryption_key_generated = False
    if 'encryption_key' not in st.session_state:
        st.session_state.encryption_key = None
    if 'timestamp_search_performed' not in st.session_state:
        st.session_state.timestamp_search_performed = False
    if 'timestamp_search_results' not in st.session_state:
        st.session_state.timestamp_search_results = []
    if 'anonymized_video_path' not in st.session_state:
        st.session_state.anonymized_video_path = None

    # Add Reset button at the top
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🔄 Reset All", type="primary", use_container_width=True):
            reset_session_state()
            st.rerun()

    # Step 1: Upload a Video
    with st.expander("Step 1: Upload a Video", expanded=True):
        uploaded_file = st.file_uploader("Click to upload", type=["mp4", "avi", "mov", "mkv"], key="upload_video")

        if uploaded_file is not None:
            st.success("Video uploaded successfully!")
            st.columns(50)
            st.video(uploaded_file, format="video/mp4")
            st.session_state.uploaded_file = uploaded_file

    # Step 2: Encryption Key Generation
    with st.expander("Step 2: Encryption Key Generation", expanded=True):
        video_name = st.text_input("Device Name", value="video1")
        # video_description = st.text_area("Video Description", value="Sample video description")

        # Display the encryption key if it has been generated
        if st.session_state.encryption_key_generated and st.session_state.encryption_key:
            st.success("Encryption key generated!")
            # st.code(st.session_state.encryption_key.decode(), language="text")

            # Provide download button
            st.download_button(
                label="⬇️ Download Encryption Key",
                data=st.session_state.encryption_key,
                file_name=f"{video_name}_encryption.key",
                mime="application/octet-stream"
            )
        else:
            if st.button("Generate Encryption Key"):
                from cryptography.fernet import Fernet
                # Generate a key and store it
                st.session_state.encryption_key = Fernet.generate_key()
                st.session_state.video_name = video_name
                st.session_state.video_description = "Sample video description"
                st.session_state.encryption_key_generated = True
                st.rerun()

    # Step 3: Encryption
    with st.expander("Step 3: Feature Extraction & Encryption", expanded=True):
        if hasattr(st.session_state, 'uploaded_file') and st.session_state.encryption_key_generated:
            if st.button("Process video",):
                with st.spinner("Processing..."):
                    try:
                        # Save the uploaded file temporarily
                        temp_video_path = os.path.join("processed_videos", st.session_state.uploaded_file.name)
                        with open(temp_video_path, "wb") as f:
                            f.write(st.session_state.uploaded_file.getbuffer())

                        # Process the video (Object Detection)
                        st.info("Processing the video...")
                        process_video(temp_video_path, "processed_videos")
                        processed_video_path = os.path.join("processed_videos",
                                                            f"{st.session_state.uploaded_file.name}_out.mp4")

                        # Create anonymized version of the processed video
                        if os.path.exists(processed_video_path):
                            st.info("Extracting features of the processed video...")
                            anonymized_path, anonymize_time = anonymize_video(processed_video_path)
                            if anonymized_path and os.path.exists(anonymized_path):
                                st.session_state.anonymized_video_path = anonymized_path
                                st.success(f"Features extracted in {anonymize_time:.2f} seconds.")
                            else:
                                st.warning("Could not create anonymized video. Will proceed with encryption.")

                            # Encrypt the processed video
                            st.info("Encrypting processed video...")
                            with open(processed_video_path, 'rb') as video_file:
                                class VideoFile:
                                    def __init__(self, file, name):
                                        self.file = file
                                        self.name = name

                                    def read(self):
                                        return self.file.read()

                                video_file_wrapper = VideoFile(video_file, os.path.basename(processed_video_path))
                                encrypted_path, key_path, encryption_time = encrypt_video(
                                    video_file_wrapper,
                                    st.session_state.video_name,
                                    st.session_state.video_description
                                )

                                # Store results in session state
                                if encrypted_path:
                                    st.session_state.encrypted_path = encrypted_path
                                    st.session_state.key_path = key_path
                                    st.session_state.encryption_time = encryption_time
                                    st.session_state.processed_video_path = processed_video_path
                                    st.session_state.encrypted_success = True

                                    st.success(f"Video encrypted in {encryption_time:.2f} seconds.")

                                    # # Display encrypted preview
                                    # with open(encrypted_path, 'rb') as f:
                                    #     encrypted_data = f.read(100)  # Just read first 100 bytes
                                    # st.code(str(encrypted_data))
                                else:
                                    st.error("Encryption failed. Please try again.")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
        else:
            st.info("Please upload a video and generate an encryption key first.")

    # Step 4: Feature Extraction (moved to its own expander)
    with st.expander("Step 4: Store Video", expanded=True):
        if hasattr(st.session_state, 'processed_video_path') and st.session_state.encrypted_success:
            if st.button("Store video"):
                with st.spinner("Processing..."):
                    # Add debug info
                    st.info("Processing video")
                    if not os.path.exists(st.session_state.processed_video_path):
                        st.error(f"Video file not found at {st.session_state.processed_video_path}")
                    else:
                        st.info(f"Video size: {os.path.getsize(st.session_state.processed_video_path)} bytes")

                    # Continue with feature extraction
                    feature_result = process_video_features(
                        st.session_state.processed_video_path,
                        st.session_state.video_name,
                        video_name_for_dir=st.session_state.video_name
                    )

                    if feature_result['status'] == 'success':
                        st.success(
                            "Video stored")
                        # st.write(f"Feature index saved to: {feature_result['index_path']}")

                        # Removed the code that displays sample frames with anomalies
                    else:
                        st.error(f"Feature extraction failed: {feature_result['message']}")
        else:
            st.info("Please encrypt a video first before extracting features.")

    # Step 5: Search and Retrieval
    with st.expander("Step 5: Search the anomalies frames", expanded=True):
        # Search options
        search_option = st.radio("Search by:", ["Video Name", "Timestamp", "Anomaly Type"], horizontal=True)

        if search_option == "Video Name":
            search_query = st.text_input("Enter video name to search")

            col1, col2 = st.columns([1, 3])
            with col1:
                search_button = st.button("Search", use_container_width=True)

            if search_button and search_query:
                search_result, retrieval_time = search_encrypted_videos(search_query)

                if search_result:
                    selected_video = st.selectbox("Select a video:", search_result)

                    if selected_video:
                        st.session_state.selected_video = selected_video

                        # Show metadata
                        metadata_path = os.path.join("metadata_key", f"{selected_video}.metadata")
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'r') as file:
                                metadata = json.load(file)
                                st.write(f"Description: {metadata['Metadata']}")
                                st.write(f"Original Extension: {metadata['original_extension']}")
                                st.write(f"Timestamp: {metadata['Timestamp']}")

                        # Show anonymized frames for the selected video
                        st.write("→ Detected frames (anonymized for security):")
                        with st.spinner("Loading anonymized frames..."):
                            from feature_extraction import get_anonymized_detected_frames
                            anonymized_frames_result = get_anonymized_detected_frames(selected_video)

                            if anonymized_frames_result['status'] == 'success' and anonymized_frames_result[
                                'frames']:
                                st.success(f"Found {len(anonymized_frames_result['frames'])} anonymized frames.")

                                # Display frames in a grid - 3 columns
                                frames = anonymized_frames_result['frames']
                                num_frames = len(frames)
                                cols_per_row = 3

                                # Process all frames
                                for i in range(0, num_frames, cols_per_row):
                                    # Create columns for each row
                                    cols = st.columns(cols_per_row)

                                    # Add frames to this row
                                    for j in range(cols_per_row):
                                        if i + j < num_frames:
                                            frame = frames[i + j]
                                            if 'frame_path' in frame and os.path.exists(frame['frame_path']):
                                                cols[j].image(frame['frame_path'],
                                                              caption=f"Frame at {frame['timestamp']}")
                            else:
                                st.warning(f"No anonymized frames available: {anonymized_frames_result['message']}")
                else:
                    st.warning("No videos found matching your search.")

        elif search_option == "Timestamp":
            # First, let the user select a video
            all_videos = get_all_encrypted_videos()

            if not all_videos:
                st.warning("No encrypted videos found.")
            else:
                selected_video = st.selectbox("Select a video:", all_videos, key="timestamp_video_select")

                if selected_video:
                    st.session_state.selected_video = selected_video

                    # Show metadata
                    metadata_path = os.path.join("metadata_key", f"{selected_video}.metadata")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as file:
                            metadata = json.load(file)
                            st.write(f"Description: {metadata['Metadata']}")
                            st.write(f"Original Extension: {metadata['original_extension']}")
                            st.write(f"Timestamp: {metadata['Timestamp']}")

                    # Add time range selection
                    st.write("Search for frames between specific timestamps:")
                    col1, col2 = st.columns(2)
                    with col1:
                        start_time = st.text_input("Start Time (HH:MM:SS)", value="00:00:00")
                    with col2:
                        end_time = st.text_input("End Time (HH:MM:SS)", value="00:10:00")

                    if st.button("Search Frames"):
                        with st.spinner("Searching for frames within the specified time range..."):
                            # Use the new helper function that includes anonymization
                            from feature_extraction import get_anonymized_frames_by_timerange

                            frames_result = get_anonymized_frames_by_timerange(
                                selected_video,
                                start_time,
                                end_time,
                                anonymize=True
                            )

                            if frames_result['status'] == 'success' and frames_result['frames']:
                                st.success(frames_result['message'])
                                st.session_state.timestamp_search_performed = True
                                st.session_state.timestamp_search_results = frames_result['frames']

                                # Display the filtered frames
                                st.write(f"Frames within the selected time range (anonymized for security):")

                                # Organize frames in a grid - 3 columns
                                filtered_frames = frames_result['frames']
                                num_frames = len(filtered_frames)
                                cols_per_row = 3

                                # Process all frames
                                for i in range(0, num_frames, cols_per_row):
                                    # Create columns for each row
                                    cols = st.columns(cols_per_row)

                                    # Add frames to this row
                                    for j in range(cols_per_row):
                                        if i + j < num_frames:
                                            frame = filtered_frames[i + j]
                                            if 'frame_path' in frame and os.path.exists(frame['frame_path']):
                                                cols[j].image(frame['frame_path'],
                                                              caption=f"Frame at {frame['timestamp']}")
                            else:
                                st.warning(frames_result['message'])
                                st.session_state.timestamp_search_performed = False
                                st.session_state.timestamp_search_results = []

        elif search_option == "Anomaly Type":
            anomaly_types = ["ACCIDENT", "SHOOTING", "VANDALISM", "EXPLOSION", "FIGHTING"]
            selected_anomalies = st.multiselect("Select anomaly types:", anomaly_types)
            # Get all videos with feature indices
            available_indices = [os.path.splitext(f)[0].replace("_index", "")
                                 for f in os.listdir("feature_indices")
                                 if f.endswith("_index.json")]

            if available_indices:
                st.success(f"Found {len(available_indices)} videos with feature indices.")
                selected_video = st.selectbox("Select a video:", available_indices)

                if selected_video:
                    st.session_state.selected_video = selected_video
            if st.button("Search"):
                                        # Show matching frames
                if selected_anomalies:
                    key_path = os.path.join("feature_indices",
                                            f"{selected_video}_feature_key.key")
                    search_result = search_video_features(
                        selected_video,
                        selected_anomalies,
                        key_path if os.path.exists(key_path) else None
                    )

                    if search_result['status'] == 'success' and search_result['frames']:
                        st.success(
                            f"Found {len(search_result['frames'])} frames with selected anomalies.")

                        # Save the search results for later use
                        st.session_state.search_frames = search_result['frames']

                        # Display ANONYMIZED frames
                        frames_with_paths = [f for f in search_result['frames'] if
                                             'frame_path' in f]
                        if frames_with_paths:
                            st.write(
                                "Frames with detected anomalies (anonymized for security):")

                            # Create directory for anonymized frames if it doesn't exist
                            anonymized_dir = os.path.join("anonymized_frames", selected_video)
                            os.makedirs(anonymized_dir, exist_ok=True)

                            # Organize frames in a grid - 3 columns
                            from anonymization import anonymize_image_file

                            # Calculate number of rows needed (ceiling division)
                            num_frames = len(frames_with_paths)
                            cols_per_row = 3

                            # Process all frames
                            for i in range(0, num_frames, cols_per_row):
                                # Create columns for each row
                                cols = st.columns(cols_per_row)

                                # Add frames to this row
                                for j in range(cols_per_row):
                                    if i + j < num_frames:
                                        frame = frames_with_paths[i + j]
                                        if os.path.exists(frame['frame_path']):
                                            # Create anonymized version of the frame
                                            frame_filename = os.path.basename(
                                                frame['frame_path'])
                                            anonymized_path = os.path.join(anonymized_dir,
                                                                           frame_filename)

                                            # Anonymize the frame if it doesn't exist already
                                            if not os.path.exists(anonymized_path):
                                                anonymized_path = anonymize_image_file(
                                                    frame['frame_path'],
                                                    anonymized_path)

                                            # Display the anonymized frame
                                            if anonymized_path and os.path.exists(
                                                    anonymized_path):
                                                cols[j].image(anonymized_path,
                                                              caption=f"Frame at {frame['timestamp']}")
                                            else:
                                                cols[j].warning(
                                                    f"Could not display anonymized frame at {frame['timestamp']}")
                    else:
                        st.info("No matching frames found for the selected anomalies.")
                else:
                    # Show all anonymized frames for the selected video when no specific anomalies are selected
                    st.write("→ All detected frames (anonymized for security):")
                    with st.spinner("Loading anonymized frames..."):
                        from feature_extraction import get_anonymized_detected_frames
                        anonymized_frames_result = get_anonymized_detected_frames(
                            selected_video)

                        if anonymized_frames_result['status'] == 'success' and \
                                anonymized_frames_result[
                                    'frames']:
                            st.success(
                                f"Found {len(anonymized_frames_result['frames'])} anonymized frames.")

                            # Display frames in a grid - 3 columns
                            frames = anonymized_frames_result['frames']
                            num_frames = len(frames)
                            cols_per_row = 3

                            # Process all frames
                            for i in range(0, num_frames, cols_per_row):
                                # Create columns for each row
                                cols = st.columns(cols_per_row)

                                # Add frames to this row
                                for j in range(cols_per_row):
                                    if i + j < num_frames:
                                        frame = frames[i + j]
                                        if 'frame_path' in frame and os.path.exists(
                                                frame['frame_path']):
                                            cols[j].image(frame['frame_path'],
                                                          caption=f"Frame at {frame['timestamp']}")
                        else:
                            st.warning(
                                f"No anonymized frames available: {anonymized_frames_result['message']}")

    # Step 6: Decryption Key
    with st.expander("Step 6: Decryption Key", expanded=True):
        if hasattr(st.session_state, 'selected_video'):
            st.info(f"Selected video: {st.session_state.selected_video}")

            key_file = st.file_uploader("Insert decryption key", type=["key"])

            if key_file is not None:
                st.session_state.key_file = key_file
                st.session_state.decryption_key_uploaded = True
                st.success("Decryption key uploaded successfully!")
        else:
            st.info("Please select a video from the search results first.")

    # Step 7: Resultant Video
    with st.expander("Step 7: Decrypted frames", expanded=True):
        if st.session_state.decryption_key_uploaded and hasattr(st.session_state, 'selected_video'):
            if not st.session_state.decryption_success:
                if st.button("Decrypt frames"):
                    with st.spinner("Decrypting video..."):
                        # Get the encrypted file
                        encrypted_file_path = os.path.join("encrypted_videos",
                                                           f"{st.session_state.selected_video}.encrypted")

                        if os.path.exists(encrypted_file_path):
                            st.info(f"Found encrypted file: {encrypted_file_path}")

                            # Debug info for the key file
                            key_content = st.session_state.key_file.read()
                            st.session_state.key_file.seek(0)  # Reset position after reading
                            st.info(f"Key file size: {len(key_content)} bytes")

                            with open(encrypted_file_path, 'rb') as encrypted_file:
                                st.info(f"Encrypted file size: {os.path.getsize(encrypted_file_path)} bytes")

                                # Perform decryption
                                decrypted_path, decryption_time = decrypt_video(
                                    encrypted_file,
                                    st.session_state.key_file,
                                    st.session_state.selected_video
                                )

                                if decrypted_path and os.path.exists(decrypted_path):
                                    st.success(f"Video decrypted successfully in {decryption_time:.2f} seconds.")
                                    st.info(f"Decrypted file saved to: {decrypted_path}")
                                    st.info(f"Decrypted file size: {os.path.getsize(decrypted_path)} bytes")

                                    # Store the path and set success flag
                                    st.session_state.decrypted_path = decrypted_path
                                    st.session_state.decryption_success = True
                                    st.rerun()
                                else:
                                    st.error("Decryption failed. Please check your key file.")
                        else:
                            st.error(f"Encrypted file not found: {encrypted_file_path}")

            # if st.session_state.decryption_success:
            #     st.write("→ Original decrypted video:")
            #
            #     try:
            #         # Use multiple methods to try displaying the video
            #         st.info(f"Attempting to play video from path: {st.session_state.decrypted_path}")
            #
            #         # Method 1: Direct path (most reliable)
            #         st.video(st.session_state.decrypted_path)
            #
            #     except Exception as e1:
            #         st.warning(f"Primary method failed: {str(e1)}")
            #         try:
            #             # Method 2: Read file and use BytesIO
            #             with open(st.session_state.decrypted_path, 'rb') as video_file:
            #                 video_bytes = video_file.read()
            #
            #             # Check for minimum video size
            #             if len(video_bytes) < 1000:
            #                 st.error(f"Video file seems too small ({len(video_bytes)} bytes) - may be corrupted")
            #             else:
            #                 # Try different approaches
            #                 st.info("Trying alternative display method...")
            #                 video_buffer = BytesIO(video_bytes)
            #
            #                 # Get file extension to determine format
            #                 _, ext = os.path.splitext(st.session_state.decrypted_path)
            #                 format_map = {
            #                     '.mp4': 'video/mp4',
            #                     '.avi': 'video/x-msvideo',
            #                     '.mov': 'video/quicktime',
            #                     '.mkv': 'video/x-matroska'
            #                 }
            #                 video_format = format_map.get(ext.lower(), 'video/mp4')
            #
            #                 st.video(video_buffer, format=video_format)
            #
            #         except Exception as e2:
            #             st.error(f"Alternative method also failed: {str(e2)}")
            #
            #             # Method 3: Provide download option as last resort
            #             st.warning("Video cannot be displayed in browser. Please download it to view:")
            #             try:
            #                 with open(st.session_state.decrypted_path, 'rb') as f:
            #                     st.download_button(
            #                         "Download video for external viewing",
            #                         f,
            #                         file_name=os.path.basename(st.session_state.decrypted_path)
            #                     )
            #             except Exception as e3:
            #                 st.error(f"Could not create download button: {str(e3)}")

                st.write("→ Detected frames (unblurred after decryption):")

                # Add debug logging
                st.info(f"Looking for frames...")

                # Check if we need to extract features
                frames_dir = os.path.join("detected_frames", st.session_state.selected_video)
                if not os.path.exists(frames_dir):
                    st.warning(f"Frames directory does not exist: {frames_dir}")
                    st.info("Extracting frames now...")

                    # Extract features from decrypted video
                    with st.spinner("Extracting features from decrypted video..."):
                        feature_result = process_video_features(
                            st.session_state.decrypted_path,
                            st.session_state.selected_video,
                            video_name_for_dir=st.session_state.selected_video  # Pass the custom name for directory
                        )
                        if feature_result['status'] == 'success':
                            st.success(
                                f"Features extracted. Found {feature_result['feature_count']} feature frames.")
                        else:
                            st.error(f"Feature extraction failed: {feature_result['message']}")

                # Get frames after ensuring directory exists - now use anonymize=False to get original frames
                frames_result = get_detected_frames(
                    st.session_state.selected_video)  # Original function for unblurred frames

                if frames_result['status'] == 'success' and frames_result['frames']:
                    st.success(f"Successfully loaded {len(frames_result['frames'])} detected frames.")

                    # Display all frames in a grid - 3 columns
                    num_frames = len(frames_result['frames'])
                    cols_per_row = 3

                    # Process all frames
                    for i in range(0, num_frames, cols_per_row):
                        # Create columns for each row
                        cols = st.columns(cols_per_row)

                        # Add frames to this row
                        for j in range(cols_per_row):
                            if i + j < num_frames:
                                frame = frames_result['frames'][i + j]
                                if 'frame_path' in frame and os.path.exists(frame['frame_path']):
                                    cols[j].image(frame['frame_path'], caption=f"Frame at {frame['timestamp']}")
                else:
                    st.warning(f"No detected frames available for this video. Message: {frames_result['message']}")

                    # As a fallback, if specific search frames were stored earlier, show those
                    if hasattr(st.session_state, 'search_frames'):
                        st.write("Showing previously found frames from search:")
                        frames_with_paths = [f for f in st.session_state.search_frames if 'frame_path' in f]
                        if frames_with_paths:
                            # Display all fallback frames in a grid - 3 columns
                            num_frames = len(frames_with_paths)
                            cols_per_row = 3

                            # Process all frames
                            for i in range(0, num_frames, cols_per_row):
                                # Create columns for each row
                                cols = st.columns(cols_per_row)

                                # Add frames to this row
                                for j in range(cols_per_row):
                                    if i + j < num_frames:
                                        frame = frames_with_paths[i + j]
                                        if os.path.exists(frame['frame_path']):
                                            cols[j].image(frame['frame_path'], caption=f"Frame at {frame['timestamp']}")
        else:
            st.info("Please upload a decryption key first.")

    # Footer
    st.divider()
    st.caption("© 2025 Video Anomaly Detection & Encryption System")


if __name__ == "__main__":
    main()
