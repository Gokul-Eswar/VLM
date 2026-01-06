import urllib.request
import os

def download_sample_videos():
    """Download sample videos for testing"""

    # Using a reliable test video URL (Big Buck Bunny)
    videos = {
        "city_traffic.mp4": "https://download.samplelib.com/mp4/sample-5s.mp4",
        # Alternative: "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_1MB.mp4"
    }

    os.makedirs("data/test_videos", exist_ok=True)

    for filename, url in videos.items():
        output_path = f"data/test_videos/{filename}"
        if not os.path.exists(output_path):
            print(f"Downloading {filename}...")
            try:
                # Add headers to avoid 403 Forbidden in some cases
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)

                urllib.request.urlretrieve(url, output_path)
                print(f"✅ Downloaded: {filename}")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
        else:
            print(f"⏭️ Already exists: {filename}")

if __name__ == "__main__":
    download_sample_videos()
    print("\n📁 Videos saved in: data/test_videos/")
