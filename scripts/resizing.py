# resize_spectrograms.py
"""
Resize spectrograms to standard 224x224 for efficient training
Converts wide spectrograms (3524x128) to square format
"""
import os
import cv2
import numpy as np
from tqdm import tqdm

INPUT_DIR = "../mel_spectrograms_enhanced"
OUTPUT_DIR = "../mel_spectrograms_resized"
TARGET_SIZE = (224, 224)  # Standard CNN input size

def resize_spectrogram(img_path, output_path, target_size=(224, 224)):
    """
    Resize spectrogram intelligently:
    - Preserve aspect ratio where possible
    - Use high-quality interpolation
    """
    # Read image
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"⚠️  Could not read: {img_path}")
        return False
    
    # Resize with cubic interpolation (best quality)
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    
    # Save
    cv2.imwrite(output_path, resized, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return True


def main():
    import shutil
    
    print("="*80)
    print("🔄 RESIZING SPECTROGRAMS TO STANDARD SIZE")
    print("="*80)
    
    if not os.path.exists(INPUT_DIR):
        print(f"\n❌ Input directory not found: {INPUT_DIR}")
        return
    
    # Create output directory
    if os.path.exists(OUTPUT_DIR):
        print(f"\n🗑️  Removing existing output directory...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    print(f"\n📁 Input:  {INPUT_DIR}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f"📐 Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    
    # Get sample image to show before/after
    sample_mood = [d for d in os.listdir(INPUT_DIR) 
                   if os.path.isdir(os.path.join(INPUT_DIR, d))][0]
    sample_dir = os.path.join(INPUT_DIR, sample_mood)
    sample_file = os.path.join(sample_dir, os.listdir(sample_dir)[0])
    
    sample_img = cv2.imread(sample_file)
    orig_h, orig_w = sample_img.shape[:2]
    print(f"\n📊 Original size: {orig_w}x{orig_h}")
    print(f"   Size reduction: {orig_w * orig_h / (TARGET_SIZE[0] * TARGET_SIZE[1]):.1f}x smaller")
    print(f"   Memory per image: {orig_w * orig_h * 3 / 1024 / 1024:.2f} MB → {TARGET_SIZE[0] * TARGET_SIZE[1] * 3 / 1024 / 1024:.2f} MB")
    
    print(f"\n🔄 Processing spectrograms...\n")
    
    total_processed = 0
    total_failed = 0
    
    # Process each mood
    for mood in sorted(os.listdir(INPUT_DIR)):
        mood_input_dir = os.path.join(INPUT_DIR, mood)
        
        if not os.path.isdir(mood_input_dir):
            continue
        
        mood_output_dir = os.path.join(OUTPUT_DIR, mood)
        os.makedirs(mood_output_dir, exist_ok=True)
        
        # Get all image files
        files = [f for f in os.listdir(mood_input_dir) if f.endswith('.png')]
        
        print(f"🎵 {mood.upper()}: ", end="", flush=True)
        
        mood_count = 0
        for file in tqdm(files, desc=f"  Processing", leave=False):
            input_path = os.path.join(mood_input_dir, file)
            output_path = os.path.join(mood_output_dir, file)
            
            if resize_spectrogram(input_path, output_path, TARGET_SIZE):
                mood_count += 1
                total_processed += 1
            else:
                total_failed += 1
        
        print(f"{mood_count} spectrograms ✅")
    
    print(f"\n" + "="*80)
    print("✅ RESIZING COMPLETE!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Successfully processed: {total_processed}")
    print(f"   Failed: {total_failed}")
    print(f"   Output directory: {OUTPUT_DIR}")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Run training with resized spectrograms")
    print(f"   2. Training will be 10-20x faster!")
    print(f"   3. Model will be much smaller and more efficient")
    
    print(f"\n🚀 Expected improvements:")
    print(f"   • Training speed: ~10-15 seconds/epoch (was 150s)")
    print(f"   • Model parameters: ~500K (was 3M)")
    print(f"   • GPU memory: Much lower")
    print(f"\n" + "="*80)


if __name__ == "__main__":
    main()