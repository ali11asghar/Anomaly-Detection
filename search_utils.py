import os
import time

def search_encrypted_videos(query):
    result = []
    start_time = time.time()

    for filename in os.listdir("encrypted_videos"):
        if filename.endswith(".encrypted") and query.lower() in filename.lower():
            result.append(os.path.splitext(filename)[0])

    return result, time.time() - start_time
