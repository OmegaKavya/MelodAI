import os
import subprocess
import re
import time
import sys
from pydub import AudioSegment
from pydub.utils import which
from songs_links import SONG_LINKS

# Ensure ffmpeg is correctly set
AudioSegment.converter = which("ffmpeg")
DATA_DIR = os.path.join("..", "data")

def clean_filename(name: str) -> str:
    """Sanitize filenames for macOS."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r'\s+', '_', name)
    return name.strip('_')

def download_audio(url, out_dir):
    """Download audio with specific options that work."""
    try:
        cmd = [
            "yt-dlp",
            "-f", "bestaudio/best",  # More flexible format selection
            "--extract-audio",
            "--audio-format", "mp3", 
            "--audio-quality", "0",
            "--no-playlist",
            "--restrict-filenames",
            "--no-warnings",  # Suppress warnings but still download
            "-o", os.path.join(out_dir, "%(title)s.%(ext)s"),
            url
        ]
        
        print(f"    Downloading...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)  # Increased timeout for longer downloads
        
        if result.returncode == 0:
            print(f"    ✅ Download completed")
            return True
        else:
            print(f"    ❌ Download failed: {result.stderr[-100:]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"    ❌ Download timeout")
        return False
    except Exception as e:
        print(f"    ❌ Unexpected error: {e}")
        return False

def trim_audio(mp3_path, duration_ms=90000, skip_ms=0):  # CHANGED: 90 seconds, no skip
    """Trim to 90s clip to capture full song sections."""
    try:
        print(f"    Loading audio file...")
        song = AudioSegment.from_file(mp3_path)
        print(f"    Original length: {len(song)//1000}s")
        
        if len(song) > duration_ms:
            # Take first 90 seconds to capture intro + verse + chorus
            song = song[:duration_ms]
            print(f"    Trimmed to: {len(song)//1000}s (first 90 seconds)")
        else:
            # If shorter than 90s, keep as is but warn
            print(f"    ⚠️  Short audio: {len(song)//1000}s (less than 90 seconds)")
        
        song.export(mp3_path, format="mp3")
        return True
    except Exception as e:
        print(f"    ❌ Trimming failed: {e}")
        return False

def find_mp3_files(out_dir):
    """Find all MP3 files in directory."""
    return [f for f in os.listdir(out_dir) if f.lower().endswith('.mp3')]

def main():
    print("🔍 Checking dependencies...")
    
    try:
        test_result = subprocess.run(
            ["yt-dlp", "--version"], 
            capture_output=True, text=True, check=True
        )
        print(f"✅ yt-dlp version: {test_result.stdout.strip()}")
    except:
        print("❌ yt-dlp not working properly")
        return
    
    if not which("ffmpeg"):
        print("❌ ffmpeg not found")
        return
    print("✅ ffmpeg is installed")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    print("\n🎧 Starting MelodAI EXTENDED Dataset Builder...")
    print("📥 Downloading 90-SECOND songs for better segmentation...\n")
    print("🎯 WHY 90 SECONDS?")
    print("   - Captures intro, verse, and chorus sections")
    print("   - Allows meaningful segmentation (3×30s segments)")
    print("   - Provides diverse audio content for training")
    print("   - Expected accuracy boost: 55-70%\n")

    total_downloaded = 0
    total_expected = sum(len(urls) for urls in SONG_LINKS.values())
    
    print(f"🎯 Target: {total_expected} songs across {len(SONG_LINKS)} moods")
    print(f"📊 Expected segments: {total_expected * 3} spectrograms\n")

    for mood, urls in SONG_LINKS.items():
        out_dir = os.path.join(DATA_DIR, mood)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"🎵 Processing: {mood.upper()} ({len(urls)} songs)")
        print(f"{'='*50}")

        # Check what files already exist
        existing_files = find_mp3_files(out_dir)
        print(f"📊 Found {len(existing_files)} existing songs")
        
        mood_downloaded = 0
        
        for i, url in enumerate(urls, 1):
            try:
                print(f"\n📥 [{mood.upper()}] {i}/{len(urls)}")
                print(f"   URL: {url}")
                
                # Count files before download
                files_before = set(find_mp3_files(out_dir))
                
                # Download the audio
                success = download_audio(url, out_dir)
                if not success:
                    continue
                
                # Wait for file system
                time.sleep(2)
                
                # Find new files by comparing before/after
                files_after = set(find_mp3_files(out_dir))
                new_files = files_after - files_before
                
                if new_files:
                    # Use the first new file found
                    new_file = list(new_files)[0]
                    mp3_path = os.path.join(out_dir, new_file)
                    print(f"   ✅ Downloaded: {new_file}")
                    
                    # Trim the audio to 90 seconds
                    print(f"   ✂️ Trimming to 90 seconds for better segmentation...")
                    if trim_audio(mp3_path, duration_ms=90000, skip_ms=0):
                        # Clean filename if needed
                        clean_name = clean_filename(os.path.splitext(new_file)[0]) + ".mp3"
                        new_clean_path = os.path.join(out_dir, clean_name)
                        
                        if mp3_path != new_clean_path:
                            os.rename(mp3_path, new_clean_path)
                            print(f"   🎉 Final: {clean_name}")
                        else:
                            print(f"   🎉 Final: {new_file}")
                        
                        mood_downloaded += 1
                        total_downloaded += 1
                    else:
                        print(f"   ❌ Failed to trim audio")
                else:
                    print(f"   ❌ No new MP3 file found after download")
                    print(f"   📁 Current files: {list(files_after)}")
                
            except KeyboardInterrupt:
                print(f"\n⚠️ Process interrupted by user")
                sys.exit(1)
            except Exception as e:
                print(f"   ❌ Unexpected error: {e}")
        
        print(f"\n📊 {mood.upper()} Complete: {mood_downloaded}/{len(urls)} songs downloaded")

    # Final summary
    print(f"\n{'='*60}")
    print("🎉 90-SECOND DOWNLOAD COMPLETE!")
    print(f"{'='*60}")
    print(f"📈 Total: {total_downloaded}/{total_expected} songs")
    print(f"📁 Location: {os.path.abspath(DATA_DIR)}")
    print(f"🎵 Audio length: 90 seconds per song")
    print(f"🔮 Next step: Generate 3 segments per song (270s total audio content)")
    
    # Show final folder structure
    print(f"\n📂 Final folder structure:")
    for mood in SONG_LINKS.keys():
        mood_dir = os.path.join(DATA_DIR, mood)
        if os.path.exists(mood_dir):
            mp3_count = len([f for f in os.listdir(mood_dir) if f.endswith('.mp3')])
            print(f"   ├── {mood}/: {mp3_count} songs (90s each)")

if __name__ == "__main__":
    main()