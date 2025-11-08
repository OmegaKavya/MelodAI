# advanced_diagnostic.py
"""
Advanced diagnostic tool to check if spectrograms are discriminative
Updated for 3-class system: energetic, peaceful, emotional
Checks multiple statistical properties beyond just mean/std
"""
import os
import numpy as np
from PIL import Image
from scipy import stats
from collections import defaultdict

SPECTROGRAM_DIR = "../mel_spectrograms_resized"

def load_spectrograms(mood_dir, max_samples=30):
    """Load spectrograms for a mood"""
    images = []
    files = sorted([f for f in os.listdir(mood_dir) if f.endswith('.png')])[:max_samples]
    
    for f in files:
        img = Image.open(os.path.join(mood_dir, f))
        img_array = np.array(img)
        images.append(img_array)
    
    return np.array(images)

def compute_advanced_stats(images):
    """Compute multiple statistical measures"""
    stats_dict = {}
    
    # Flatten all images
    flat = images.reshape(len(images), -1)
    
    # Basic stats
    stats_dict['mean'] = np.mean(flat, axis=1)
    stats_dict['std'] = np.std(flat, axis=1)
    stats_dict['median'] = np.median(flat, axis=1)
    
    # Per-channel stats (RGB)
    for c, color in enumerate(['R', 'G', 'B']):
        channel_data = images[:, :, :, c].reshape(len(images), -1)
        stats_dict[f'{color}_mean'] = np.mean(channel_data, axis=1)
        stats_dict[f'{color}_std'] = np.std(channel_data, axis=1)
        stats_dict[f'{color}_energy'] = np.sum(channel_data ** 2, axis=1)
    
    # Distribution properties
    stats_dict['skewness'] = stats.skew(flat, axis=1)
    stats_dict['kurtosis'] = stats.kurtosis(flat, axis=1)
    
    # Energy metrics
    stats_dict['total_energy'] = np.sum(flat ** 2, axis=1)
    stats_dict['entropy'] = -np.sum(flat * np.log(flat + 1e-10), axis=1)
    
    # High/Low frequency proxy (top vs bottom half)
    mid_point = images.shape[1] // 2
    top_half = images[:, :mid_point, :, :].reshape(len(images), -1)
    bottom_half = images[:, mid_point:, :, :].reshape(len(images), -1)
    stats_dict['high_freq_energy'] = np.mean(top_half, axis=1)
    stats_dict['low_freq_energy'] = np.mean(bottom_half, axis=1)
    
    # Temporal variation (left vs right, beginning vs end)
    time_mid = images.shape[2] // 2
    first_half = images[:, :, :time_mid, :].reshape(len(images), -1)
    second_half = images[:, :, time_mid:, :].reshape(len(images), -1)
    stats_dict['temporal_variation'] = np.abs(np.mean(first_half, axis=1) - np.mean(second_half, axis=1))
    
    return stats_dict

def analyze_discriminability(mood_stats):
    """Analyze how well features separate moods"""
    print("\n" + "="*80)
    print("📊 DISCRIMINABILITY ANALYSIS")
    print("="*80)
    
    # Get all feature names
    feature_names = list(next(iter(mood_stats.values())).keys())
    
    print("\n🔍 Feature Ranges Across 3 Moods:\n")
    
    discriminative_features = []
    
    for feature in feature_names:
        values = {}
        for mood, stats_dict in mood_stats.items():
            values[mood] = stats_dict[feature]
        
        # Compute range of means across moods
        mood_means = {mood: np.mean(vals) for mood, vals in values.items()}
        min_mean = min(mood_means.values())
        max_mean = max(mood_means.values())
        range_ratio = (max_mean - min_mean) / (min_mean + 1e-8) * 100
        
        # Compute coefficient of variation across moods
        all_means = list(mood_means.values())
        cv = np.std(all_means) / (np.mean(all_means) + 1e-8) * 100
        
        # Feature is discriminative if it varies significantly across moods
        # Lower thresholds for 3-class (should be easier to separate)
        if range_ratio > 4 or cv > 2.5:  # Adjusted thresholds for 3 classes
            discriminative_features.append((feature, range_ratio, cv))
            marker = "✨"
        else:
            marker = "  "
        
        print(f"{marker} {feature:20s} | Range: {range_ratio:6.2f}% | CV: {cv:5.2f}%")
        
        # Show mood values if discriminative
        if marker == "✨":
            print(f"   {'':20s} |", end="")
            for mood in sorted(mood_means.keys()):
                print(f" {mood:10s}: {mood_means[mood]:7.2f}", end="")
            print()
    
    print("\n" + "-"*80)
    print(f"\n🎯 Discriminative Features Found: {len(discriminative_features)}/{len(feature_names)}")
    
    if len(discriminative_features) > 0:
        print("\n✅ TOP DISCRIMINATIVE FEATURES:")
        discriminative_features.sort(key=lambda x: x[2], reverse=True)
        for feat, range_r, cv in discriminative_features[:5]:
            print(f"   • {feat:20s} (CV: {cv:5.2f}%)")
    else:
        print("\n❌ NO STRONGLY DISCRIMINATIVE FEATURES FOUND!")
        print("   Spectrograms may be too similar for effective classification.")
    
    return len(discriminative_features)

def mood_separation_score(mood_stats):
    """Compute overall separation between 3 moods"""
    print("\n" + "="*80)
    print("📏 3-CLASS MOOD SEPARATION SCORES")
    print("="*80)
    
    moods = list(mood_stats.keys())
    
    # For each pair of moods, compute separability
    print("\n🔬 Pairwise Mood Distances (higher = more separable):\n")
    
    separation_matrix = defaultdict(dict)
    
    for i, mood1 in enumerate(moods):
        for j, mood2 in enumerate(moods):
            if i >= j:
                continue
            
            # Compare distributions using multiple features
            total_distance = 0
            
            for feature in ['mean', 'R_energy', 'G_energy', 'B_energy', 
                          'high_freq_energy', 'low_freq_energy', 'temporal_variation']:
                if feature in mood_stats[mood1]:
                    vals1 = mood_stats[mood1][feature]
                    vals2 = mood_stats[mood2][feature]
                    
                    # Compute normalized distance between distributions
                    mean_diff = abs(np.mean(vals1) - np.mean(vals2))
                    pooled_std = np.sqrt((np.std(vals1)**2 + np.std(vals2)**2) / 2)
                    
                    # Cohen's d (effect size)
                    if pooled_std > 0:
                        distance = mean_diff / pooled_std
                        total_distance += distance
            
            separation_matrix[mood1][mood2] = total_distance
            separation_matrix[mood2][mood1] = total_distance
            
            # Interpretation - adjusted for 3-class
            if total_distance > 4:
                marker = "✅ EXCELLENT"
            elif total_distance > 2:
                marker = "⚠️  MODERATE"
            else:
                marker = "❌ POOR"
            
            print(f"{marker}  {mood1:10s} ↔ {mood2:10s}: {total_distance:6.2f}")
    
    # Overall score
    all_distances = []
    for mood1 in separation_matrix:
        for mood2 in separation_matrix[mood1]:
            if mood1 != mood2:
                all_distances.append(separation_matrix[mood1][mood2])
    
    avg_separation = np.mean(all_distances)
    
    print(f"\n📊 Average Separation Score: {avg_separation:.2f}")
    
    # Adjusted thresholds for 3-class
    if avg_separation > 4:
        print("   ✅ EXCELLENT - 3 moods are well separated!")
    elif avg_separation > 2.5:
        print("   ⚠️  MODERATE - Some overlap between moods")
    else:
        print("   ❌ POOR - Moods are not well separated")
    
    return avg_separation

def analyze_class_characteristics(mood_stats):
    """Analyze specific characteristics of each mood class"""
    print("\n" + "="*80)
    print("🎵 3-CLASS MOOD CHARACTERISTICS")
    print("="*80)
    
    print("\n🔍 Expected Patterns:")
    print("   • Energetic: High frequency energy, high contrast, bright colors")
    print("   • Peaceful:  Balanced spectrum, smooth transitions, softer colors") 
    print("   • Emotional: Dynamic range, expressive variations, rich textures")
    
    print("\n📊 Measured Characteristics:\n")
    
    for mood, stats_dict in mood_stats.items():
        print(f"🎵 {mood.upper():10s}:")
        
        # Key metrics for each mood
        brightness = np.mean(stats_dict['mean'])
        contrast = np.mean(stats_dict['std'])
        color_vibrancy = np.mean(stats_dict['R_energy'] + stats_dict['G_energy'] + stats_dict['B_energy']) / 3
        high_freq = np.mean(stats_dict['high_freq_energy'])
        low_freq = np.mean(stats_dict['low_freq_energy'])
        dynamics = np.mean(stats_dict['temporal_variation'])
        
        print(f"   • Brightness:    {brightness:7.2f}")
        print(f"   • Contrast:      {contrast:7.2f}")
        print(f"   • Color Energy:  {color_vibrancy:7.2f}")
        print(f"   • High Freq:     {high_freq:7.2f}")
        print(f"   • Low Freq:      {low_freq:7.2f}")
        print(f"   • Dynamics:      {dynamics:7.2f}")
        
        # Character assessment
        if mood == 'energetic':
            if high_freq > low_freq and contrast > 40:
                print("   ✅ Matches expected energetic pattern")
            else:
                print("   ⚠️  Doesn't match typical energetic pattern")
                
        elif mood == 'peaceful':
            if abs(high_freq - low_freq) < 20 and contrast < 50:
                print("   ✅ Matches expected peaceful pattern")
            else:
                print("   ⚠️  Doesn't match typical peaceful pattern")
                
        elif mood == 'emotional':
            if dynamics > 15 and color_vibrancy > 1000:
                print("   ✅ Matches expected emotional pattern")
            else:
                print("   ⚠️  Doesn't match typical emotional pattern")
        
        print()

def main():
    print("\n" + "="*80)
    print("🔬 ADVANCED 3-CLASS SPECTROGRAM DISCRIMINABILITY ANALYSIS")
    print("="*80)
    print(f"\n📁 Analyzing: {os.path.abspath(SPECTROGRAM_DIR)}\n")
    
    # Load spectrograms for each mood
    mood_stats = {}
    expected_moods = ['energetic', 'peaceful', 'emotional']
    
    for mood in sorted(os.listdir(SPECTROGRAM_DIR)):
        mood_dir = os.path.join(SPECTROGRAM_DIR, mood)
        if not os.path.isdir(mood_dir) or mood not in expected_moods:
            continue
        
        print(f"📊 Loading {mood.upper()}...", end=" ")
        images = load_spectrograms(mood_dir, max_samples=30)
        print(f"{len(images)} samples")
        
        mood_stats[mood] = compute_advanced_stats(images)
    
    # Check if we have all 3 classes
    if len(mood_stats) != 3:
        print(f"\n❌ Expected 3 classes but found {len(mood_stats)}: {list(mood_stats.keys())}")
        print("   Please ensure your dataset has: energetic, peaceful, emotional")
        return
    
    # Analyze discriminability
    n_discriminative = analyze_discriminability(mood_stats)
    
    # Compute separation scores
    separation_score = mood_separation_score(mood_stats)
    
    # Analyze class characteristics
    analyze_class_characteristics(mood_stats)
    
    # Final recommendation
    print("\n" + "="*80)
    print("💡 3-CLASS RECOMMENDATIONS")
    print("="*80)
    
    # Adjusted expectations for 3-class
    if separation_score > 4 and n_discriminative > 6:
        print("\n✅ Spectrograms look EXCELLENT for 3-class classification!")
        print("   • Proceed with CNN training")
        print("   • Expected accuracy: 70-85%+")
        print("   • Model should learn patterns effectively")
        
    elif separation_score > 2.5 or n_discriminative > 4:
        print("\n⚠️  Spectrograms show MODERATE discriminability")
        print("   • Training should work with reasonable accuracy (55-75%)")
        print("   • Consider:")
        print("     - Using data augmentation")
        print("     - Try residual CNN architecture")
        print("     - Fine-tune learning rate")
        
    else:
        print("\n❌ Spectrograms show POOR discriminability")
        print("   • Current approach may struggle")
        print("   • Recommendations:")
        print("     - Check audio preprocessing parameters")
        print("     - Verify mood labels are accurate")
        print("     - Try different spectrogram types (MFCC, chroma)")
        print("     - Consider ensemble methods")
    
    print(f"\n📈 Key Metrics Summary:")
    print(f"   • Separation Score: {separation_score:.2f}/4.0+ (target)")
    print(f"   • Discriminative Features: {n_discriminative}/15+ (target)")
    print(f"   • Classes Analyzed: {len(mood_stats)}/3")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()