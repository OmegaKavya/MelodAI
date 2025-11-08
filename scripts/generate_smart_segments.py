# generate_final_hybrid_spectrograms.py
"""
FINAL HYBRID SYSTEM: Spectrograms + Data Augmentation

Given diagnostic results:
- Overall score: 3.88 (MODERATE)
- Major issues: romantic↔sad (0.63), calm↔drive (1.06)

Strategy:
1. Keep current spectrograms (they're decent for party/sad/calm separation)
2. Add data augmentation to increase diversity and implicit feature learning
3. Generate longer segments (45s instead of 30s) to capture more context
4. Create multiple augmented versions per song
"""
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

AUDIO_DIR = "../data"
OUTPUT_DIR = "../mel_spectrograms_enhanced"

# Augmentation settings
AUGMENTATIONS = {
    'original': {},
    'pitch_up': {'pitch_shift': 2},      # +2 semitones
    'pitch_down': {'pitch_shift': -2},    # -2 semitones
    'time_stretch': {'time_stretch': 1.1}, # 10% faster
}


def augment_audio(y, sr, aug_type='original', **kwargs):
    """Apply audio augmentation"""
    if aug_type == 'original':
        return y
    
    y_aug = y.copy()
    
    if 'pitch_shift' in kwargs:
        y_aug = librosa.effects.pitch_shift(
            y_aug, sr=sr, n_steps=kwargs['pitch_shift']
        )
    
    if 'time_stretch' in kwargs:
        y_aug = librosa.effects.time_stretch(
            y_aug, rate=kwargs['time_stretch']
        )
    
    return y_aug


def create_hybrid_spectrogram(y, sr, save_path):
    """
    Hybrid approach combining best features from diagnostics
    Focus on discriminative features: B_mean, G_mean, R_std
    """
    try:
        # MEL SPECTROGRAM
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=128,
            hop_length=256,
            n_fft=2048,
            fmin=20,
            fmax=8000
        )
        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)
        
        # HARMONIC-PERCUSSIVE SEPARATION
        harmonic, percussive = librosa.effects.hpss(y)
        
        mel_harmonic = librosa.feature.melspectrogram(
            y=harmonic, sr=sr, n_mels=128, hop_length=256
        )
        mel_harmonic_db = librosa.power_to_db(mel_harmonic, ref=np.max)
        
        mel_percussive = librosa.feature.melspectrogram(
            y=percussive, sr=sr, n_mels=128, hop_length=256
        )
        mel_percussive_db = librosa.power_to_db(mel_percussive, ref=np.max)
        
        # CHROMA (harmonic content)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=256)
        
        # SPECTRAL CONTRAST
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=256)
        
        # Align time dimensions
        min_time = min(
            mel_db.shape[1],
            mel_harmonic_db.shape[1],
            chroma.shape[1],
            contrast.shape[1]
        )
        
        mel_db = mel_db[:, :min_time]
        mel_harmonic_db = mel_harmonic_db[:, :min_time]
        mel_percussive_db = mel_percussive_db[:, :min_time]
        chroma = chroma[:, :min_time]
        contrast = contrast[:, :min_time]
        
        # Normalize function
        def norm(x):
            x_min, x_max = np.percentile(x, [2, 98])
            return np.clip((x - x_min) / (x_max - x_min + 1e-8), 0, 1)
        
        # RED: Percussive content (rhythm/beats)
        # High for party/drive, low for calm/romantic/sad
        percussive_norm = norm(mel_percussive_db)
        
        # GREEN: Harmonic content (melody/harmony)
        # Different patterns for romantic vs sad
        harmonic_norm = norm(mel_harmonic_db)
        
        # Expand chroma
        chroma_norm = norm(chroma)
        chroma_expanded = np.zeros((128, min_time))
        for i in range(12):
            start = i * 10
            end = min(start + 11, 128)
            chroma_expanded[start:end, :] = chroma_norm[i:i+1, :]
        
        # BLUE: Overall energy + spectral texture
        mel_norm = norm(mel_db)
        contrast_norm = norm(contrast)
        contrast_resized = np.resize(contrast_norm, (128, min_time))
        
        # Combine channels
        red_channel = percussive_norm
        
        green_channel = (
            harmonic_norm * 0.5 +
            chroma_expanded * 0.5
        )
        
        blue_channel = (
            mel_norm * 0.6 +
            contrast_resized * 0.4
        )
        
        # Stack
        img = np.stack([red_channel, green_channel, blue_channel], axis=-1)
        img = np.clip(img, 0, 1)
        
        # Light sharpening on boundaries
        for c in range(3):
            img[..., c] = gaussian_filter(img[..., c], sigma=0.5)
        
        # Save
        plt.imsave(save_path, img, origin='lower', dpi=300)
        return True
        
    except Exception as e:
        print(f"⚠️  {e}")
        return False


def main():
    import shutil
    
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("🎯 FINAL HYBRID SYSTEM: Spectrograms + Data Augmentation")
    print("="*80)
    print("\n🔬 Strategy:")
    print("   1. Harmonic-Percussive Separation (HPSS)")
    print("   2. 45-second segments (more context)")
    print("   3. Data augmentation (pitch shift, time stretch)")
    print("   4. Multiple versions per song → 2x-3x more training data")
    print("\n📊 Expected output: ~1200+ spectrograms (vs 443 currently)")
    print("\n" + "="*80 + "\n")
    
    total_saved = 0
    
    for mood in sorted(os.listdir(AUDIO_DIR)):
        mood_dir = os.path.join(AUDIO_DIR, mood)
        if not os.path.isdir(mood_dir):
            continue
        
        save_dir = os.path.join(OUTPUT_DIR, mood)
        os.makedirs(save_dir, exist_ok=True)
        
        files = sorted([f for f in os.listdir(mood_dir) if f.endswith(".mp3")])
        print(f"🎵 {mood.upper()} ({len(files)} songs):", end=" ", flush=True)
        
        mood_count = 0
        
        for file in files:
            path = os.path.join(mood_dir, file)
            
            try:
                # Load full 90 seconds
                y, sr = librosa.load(path, sr=22050, duration=90, mono=True)
                
                # Create 2 segments: 0-45s and 45-90s (longer context)
                segments = [
                    (0, 45, 'seg1'),
                    (45, 90, 'seg2')
                ]
                
                for start, end, seg_name in segments:
                    seg = y[start*sr : end*sr]
                    
                    if len(seg) < 45 * sr:
                        continue
                    
                    # Apply augmentations
                    for aug_name, aug_params in AUGMENTATIONS.items():
                        y_aug = augment_audio(seg, sr, aug_name, **aug_params)
                        
                        base_name = os.path.splitext(file)[0]
                        filename = f"{base_name}_{seg_name}_{aug_name}.png"
                        save_path = os.path.join(save_dir, filename)
                        
                        if create_hybrid_spectrogram(y_aug, sr, save_path):
                            mood_count += 1
                            total_saved += 1
                
            except Exception as e:
                print("❌", end="", flush=True)
                continue
        
        print(f"{mood_count} spectrograms ✅")
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"\n📈 Total Generated: {total_saved} spectrograms")
    print(f"📁 Location: {OUTPUT_DIR}")
    print("\n💡 Benefits of this approach:")
    print("   ✅ 2-3x more training data (augmentation)")
    print("   ✅ Longer segments (45s) = more context")
    print("   ✅ HPSS separates rhythm from harmony")
    print("   ✅ Better generalization from pitch/tempo variations")
    print("\n📊 Expected CNN Training Results:")
    print("   • With current spectrograms (3.88 separation): 45-55% accuracy")
    print("   • With augmentation + deeper CNN: 55-70% accuracy")
    print("   • With transfer learning: 65-80% accuracy")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()