# spectrogram_comparison.py
"""
Creates comparison visualizations of spectrograms for documentation
Shows why peaceful vs emotional are hard to separate
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import random

def create_spectrogram_comparison():
    """Creates side-by-side comparison of spectrograms from all 3 classes"""
    
    # Get sample images from each class
    classes = ['energetic', 'peaceful', 'emotional']
    samples = {}
    
    print("📊 Collecting sample spectrograms...")
    
    for class_name in classes:
        class_dir = f"../mel_spectrograms_resized/{class_name}"
        if not os.path.exists(class_dir):
            print(f"❌ Directory not found: {class_dir}")
            continue
            
        files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
        if not files:
            print(f"❌ No images found in {class_dir}")
            continue
            
        # Pick 3 random samples from each class
        selected_files = random.sample(files, min(3, len(files)))
        samples[class_name] = []
        
        for file in selected_files:
            try:
                img_path = os.path.join(class_dir, file)
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img) / 255.0
                samples[class_name].append(img_array)
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
                continue
    
    # Create comparison figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Spectrogram Comparison: Why Peaceful vs Emotional is Challenging', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Plot samples
    for i, class_name in enumerate(classes):
        for j in range(3):
            if j < len(samples[class_name]):
                axes[i, j].imshow(samples[class_name][j])
                axes[i, j].set_title(f'{class_name.title()} - Sample {j+1}', 
                                   fontweight='bold', fontsize=12)
            else:
                axes[i, j].axis('off')
            
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].set_xlabel('Time →', fontsize=9)
            axes[i, j].set_ylabel('Frequency →', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('../models/spectrogram_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved spectrogram comparison: ../models/spectrogram_comparison.png")
    
    # Create feature analysis plot
    create_feature_analysis(samples)
    
    return samples

def create_feature_analysis(samples):
    """Creates analysis of why classes are similar/different"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Acoustic Feature Analysis: Energetic vs Peaceful vs Emotional', 
                 fontsize=14, fontweight='bold')
    
    # Analyze brightness/energy distribution
    brightness_data = {cls: [] for cls in samples}
    contrast_data = {cls: [] for cls in samples}
    
    for class_name, images in samples.items():
        for img in images:
            # Brightness (mean pixel value)
            brightness = np.mean(img)
            brightness_data[class_name].append(brightness)
            
            # Contrast (standard deviation)
            contrast = np.std(img)
            contrast_data[class_name].append(contrast)
    
    # Plot 1: Brightness comparison
    positions = [1, 2, 3]
    colors = ['red', 'blue', 'green']
    
    for i, (class_name, brightness_vals) in enumerate(brightness_data.items()):
        axes[0, 0].boxplot(brightness_vals, positions=[positions[i]], 
                          patch_artist=True, 
                          boxprops=dict(facecolor=colors[i], alpha=0.7),
                          labels=[class_name.title()])
    
    axes[0, 0].set_title('Average Brightness Comparison')
    axes[0, 0].set_ylabel('Brightness (0-1 scale)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Contrast comparison
    for i, (class_name, contrast_vals) in enumerate(contrast_data.items()):
        axes[0, 1].boxplot(contrast_vals, positions=[positions[i]], 
                          patch_artist=True,
                          boxprops=dict(facecolor=colors[i], alpha=0.7),
                          labels=[class_name.title()])
    
    axes[0, 1].set_title('Contrast/Variability Comparison')
    axes[0, 1].set_ylabel('Contrast (std dev)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: RGB channel analysis for one sample each
    axes[1, 0].set_title('Color Distribution (RGB Channels)')
    for i, class_name in enumerate(samples):
        if samples[class_name]:
            img = samples[class_name][0]  # Use first sample
            for channel, color, label in zip([0, 1, 2], ['red', 'green', 'blue'], ['R', 'G', 'B']):
                channel_data = img[:, :, channel].flatten()
                axes[1, 0].hist(channel_data, bins=50, alpha=0.6, 
                               color=color, label=f'{class_name} {label}', 
                               density=True)
    
    axes[1, 0].set_xlabel('Pixel Intensity')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Separation challenge explanation
    axes[1, 1].text(0.1, 0.8, '🔍 Separation Analysis', fontsize=12, fontweight='bold')
    axes[1, 1].text(0.1, 0.6, '• Energetic: High contrast, bright colors\n  (Easy to identify)', fontsize=10)
    axes[1, 1].text(0.1, 0.4, '• Peaceful/Emotional: Similar patterns\n  (Hard to separate)', fontsize=10)
    axes[1, 1].text(0.1, 0.2, f'• Separation score: 0.007\n  (Extremely low)', fontsize=10, color='red')
    axes[1, 1].text(0.1, 0.05, 'Conclusion: Acoustic features alone\ncannot distinguish peaceful vs emotional', 
                   fontsize=10, style='italic', color='darkred')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].spines['right'].set_visible(False)
    axes[1, 1].spines['bottom'].set_visible(False)
    axes[1, 1].spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('../models/feature_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved feature analysis: ../models/feature_analysis.png")

def create_performance_summary():
    """Creates a performance summary visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Performance data from your evaluation
    classes = ['Energetic', 'Emotional', 'Peaceful']
    f1_scores = [0.88, 0.74, 0.27]  # From your 66% model
    colors = ['green', 'orange', 'red']
    
    bars = ax.bar(classes, f1_scores, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.0%}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('F1-Score', fontweight='bold')
    ax.set_title('Model Performance by Mood Class\n(66% Overall Accuracy)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotations
    ax.text(0.5, 0.85, '✅ Excellent separation', ha='center', transform=ax.transAxes, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax.text(0.5, 0.75, '⚠️  Good but room for improvement', ha='center', transform=ax.transAxes,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    ax.text(0.5, 0.65, '❌ Fundamental separation challenge', ha='center', transform=ax.transAxes,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    plt.tight_layout()
    plt.savefig('../models/performance_summary.png', dpi=300, bbox_inches='tight')
    print("✅ Saved performance summary: ../models/performance_summary.png")

if __name__ == "__main__":
    print("🎵 Generating spectrogram comparison visualizations...")
    print("=" * 60)
    
    # Create all visualizations
    samples = create_spectrogram_comparison()
    create_performance_summary()
    
    print("=" * 60)
    print("📊 All visualizations generated:")
    print("   • ../models/spectrogram_comparison.png")
    print("   • ../models/feature_analysis.png") 
    print("   • ../models/performance_summary.png")
    print("\n💡 Use these in your GitHub documentation to show:")
    print("   - Why the 3-class problem is challenging")
    print("   - Where the model succeeds and struggles")
    print("   - The fundamental limitations of acoustic features")