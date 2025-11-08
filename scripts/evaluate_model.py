# evaluate_model.py
"""
Comprehensive model evaluation with confusion matrix and per-class analysis
Updated for 3-class system: energetic, peaceful, emotional
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

def plot_confusion_matrix(cm, class_names, save_path='../models/confusion_matrix.png'):
    """Plot confusion matrix with percentages"""
    fig, ax = plt.subplots(figsize=(10, 8))  # Smaller for 3 classes
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations (count + percentage)
    annotations = np.array([[f'{count}\n({pct:.1f}%)' 
                           for count, pct in zip(row_counts, row_pcts)]
                          for row_counts, row_pcts in zip(cm, cm_percent)])
    
    # Plot heatmap
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'},
                linewidths=0.5, linecolor='gray', ax=ax)
    
    plt.title('3-Class Confusion Matrix\n(Actual vs Predicted)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrix saved: {save_path}")


def analyze_misclassifications(y_true, y_pred, class_names):
    """Analyze which moods are most confused"""
    print("\n" + "="*80)
    print("🔍 3-CLASS MISCLASSIFICATION ANALYSIS")
    print("="*80)
    
    # Get confusion pairs
    confusion_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                count = np.sum((y_true == i) & (y_pred == j))
                if count > 0:
                    confusion_pairs.append((class_names[i], class_names[j], count))
    
    # Sort by count
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("\n📉 Most Common Confusions:")
    for i, (true_class, pred_class, count) in enumerate(confusion_pairs[:6], 1):  # Fewer for 3 classes
        percentage = count / np.sum(y_true == class_names.index(true_class)) * 100
        print(f"   {i:2d}. {true_class:10s} → {pred_class:10s}: {count:3d} samples ({percentage:.1f}%)")
    
    # Expected confusion patterns for 3-class system
    print("\n🎵 Expected Confusion Patterns:")
    expected_pairs = [
        ('energetic', 'emotional', "Both can have high energy"),
        ('peaceful', 'emotional', "Both can have expressive dynamics"),
        ('energetic', 'peaceful', "Should be well separated")
    ]
    
    for mood1, mood2, reason in expected_pairs:
        if mood1 in class_names and mood2 in class_names:
            idx1 = class_names.index(mood1)
            idx2 = class_names.index(mood2)
            
            conf12 = np.sum((y_true == idx1) & (y_pred == idx2))
            conf21 = np.sum((y_true == idx2) & (y_pred == idx1))
            total = conf12 + conf21
            
            if total > 0:
                marker = "⚠️" if total > len(y_true) * 0.1 else "✅"  # >10% is concerning
                print(f"   {marker} {mood1} ↔ {mood2}: {total} confusions")
                print(f"        Reason: {reason}")
                if conf12 > 0:
                    print(f"        - {mood1}→{mood2}: {conf12}")
                if conf21 > 0:
                    print(f"        - {mood2}→{mood1}: {conf21}")


def evaluate_model(model_path, spectrogram_dir):
    """Comprehensive model evaluation for 3-class system"""
    print("="*80)
    print("🧪 3-CLASS MODEL EVALUATION")
    print("="*80)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        print("   Try training first: python3 train_optimized_model.py")
        return
    
    print(f"\n📁 Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print(f"📁 Loading spectrograms: {spectrogram_dir}")
    
    # Auto-detect image size
    sample_mood = [d for d in os.listdir(spectrogram_dir) 
                   if os.path.isdir(os.path.join(spectrogram_dir, d))][0]
    sample_dir = os.path.join(spectrogram_dir, sample_mood)
    sample_file = os.path.join(sample_dir, os.listdir(sample_dir)[0])
    
    try:
        from PIL import Image
        with Image.open(sample_file) as img:
            target_size = img.size
    except:
        target_size = (224, 224)
    
    # Create validation generator (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    val_gen = val_datagen.flow_from_directory(
        spectrogram_dir,
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        color_mode='rgb'
    )
    
    print(f"\n📊 Evaluation Dataset:")
    print(f"   Samples: {val_gen.samples}")
    print(f"   Classes: {list(val_gen.class_indices.keys())}")
    
    # Check if we have 3 classes
    if len(val_gen.class_indices) != 3:
        print(f"⚠️  Expected 3 classes but found {len(val_gen.class_indices)}")
        print("   Ensure your dataset has: energetic, peaceful, emotional")
    
    # Get predictions
    print(f"\n🔮 Making predictions...")
    predictions = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes
    
    # Get class names
    class_names = list(val_gen.class_indices.keys())
    
    # Overall metrics
    print("\n" + "="*80)
    print("📈 OVERALL METRICS")
    print("="*80)
    
    overall_acc = np.mean(y_pred == y_true)
    print(f"\n✅ Overall Accuracy: {overall_acc:.2%}")
    
    # Per-class metrics
    print("\n" + "="*80)
    print("📊 PER-CLASS PERFORMANCE")
    print("="*80)
    
    print(f"\n{'Mood':10s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>8s}")
    print("-" * 55)
    
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )
    
    for i, name in enumerate(class_names):
        print(f"{name:10s} {precision[i]:>9.2%} {recall[i]:>9.2%} {f1[i]:>9.2%} {support[i]:>8d}")
    
    # Weighted averages
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    print("-" * 55)
    print(f"{'WEIGHTED':10s} {precision_avg:>9.2%} {recall_avg:>9.2%} {f1_avg:>9.2%}")
    
    # Classification report
    print("\n" + "="*80)
    print("📋 DETAILED CLASSIFICATION REPORT")
    print("="*80)
    print()
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names)
    
    # Misclassification analysis
    analyze_misclassifications(y_true, y_pred, class_names)
    
    # Identify best and worst performing classes
    print("\n" + "="*80)
    print("🏆 BEST & WORST PERFORMING MOODS")
    print("="*80)
    
    f1_scores = list(zip(class_names, f1))
    f1_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n✅ Best Performing:")
    for name, score in f1_scores[:2]:  # Top 2 for 3 classes
        print(f"   • {name:10s}: {score:.2%} F1-score")
    
    print(f"\n⚠️  Worst Performing:")
    for name, score in f1_scores[-1:]:  # Only worst for 3 classes
        print(f"   • {name:10s}: {score:.2%} F1-score")
    
    # Class balance analysis
    print(f"\n📊 Class Distribution in Validation Set:")
    for name, count in zip(class_names, support):
        percentage = count / len(y_true) * 100
        print(f"   • {name:10s}: {count:3d} samples ({percentage:.1f}%)")
    
    # Recommendations - adjusted for 3-class
    print("\n" + "="*80)
    print("💡 3-CLASS RECOMMENDATIONS")
    print("="*80)
    
    # Adjusted thresholds for 3-class (should be higher)
    if overall_acc >= 0.80:
        print("\n🌟 EXCELLENT! Model performs exceptionally well for 3-class!")
        print("   • Production-ready")
        print("   • Consider deploying for real use")
        
    elif overall_acc >= 0.70:
        print("\n✅ VERY GOOD! Solid 3-class performance.")
        print("   • Suitable for most applications")
        print("   • Minor tuning could improve further")
        
    elif overall_acc >= 0.60:
        print("\n📊 GOOD! Reasonable performance.")
        print("   • Consider:")
        print("     - Fine-tuning hyperparameters")
        print("     - Adding more training data for weak classes")
        print("     - Trying different architectures")
        
    elif overall_acc >= 0.50:
        print("\n⚠️  MODERATE performance needs improvement.")
        print("   • Try:")
        print("     - More data augmentation")
        print("     - Transfer learning")
        print("     - Longer training with early stopping")
        
    else:
        print("\n❌ LOW performance - needs significant improvement.")
        print("   • Strongly consider:")
        print("     - Regenerating spectrograms")
        print("     - Using pre-trained models")
        print("     - Checking data quality and labels")
    
    # Check for specific 3-class issues
    print(f"\n🔍 3-Class Specific Analysis:")
    
    # Check if any class has very low performance
    weak_classes = [(name, f1_score) for name, f1_score in zip(class_names, f1) if f1_score < 0.5]
    if weak_classes:
        print(f"   ⚠️  Weak classes (<50% F1):")
        for name, score in weak_classes:
            print(f"      • {name}: {score:.2%}")
        print("   • Focus on improving these classes with targeted augmentation")
    
    # Check balance
    max_support = max(support)
    min_support = min(support)
    imbalance_ratio = max_support / min_support if min_support > 0 else float('inf')
    
    if imbalance_ratio > 2:
        print(f"   ⚠️  Class imbalance detected (ratio: {imbalance_ratio:.1f})")
        print("   • Consider using class weights in training")
    
    # Confidence analysis
    print(f"\n🎯 Confidence Analysis:")
    max_probs = np.max(predictions, axis=1)
    avg_confidence = np.mean(max_probs)
    low_confidence = np.sum(max_probs < 0.7) / len(max_probs) * 100
    
    print(f"   • Average prediction confidence: {avg_confidence:.2%}")
    print(f"   • Samples with low confidence (<70%): {low_confidence:.1f}%")
    
    if avg_confidence < 0.7:
        print("   ⚠️  Model is uncertain about many predictions")
    
    print("\n" + "="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained 3-class mood classification model')
    parser.add_argument('--model', type=str, default='../models/best_model.h5',
                       help='Path to trained model')
    parser.add_argument('--data', type=str, default='../mel_spectrograms_resized',
                       help='Path to spectrogram directory')
    
    args = parser.parse_args()
    
    # Check for data directory
    if not os.path.exists(args.data):
        # Try alternatives
        alternatives = ['../mel_spectrograms_fusion', '../mel_spectrograms_enhanced']
        for alt in alternatives:
            if os.path.exists(alt):
                args.data = alt
                print(f"📁 Using alternative dataset: {alt}")
                break
        else:
            print("❌ No spectrogram directory found!")
            print("   Please generate spectrograms first")
            return
    
    evaluate_model(args.model, args.data)


if __name__ == "__main__":
    main()