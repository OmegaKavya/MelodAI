# train_optimized_model.py
"""
Optimized Training Pipeline for 3-Class Mood Classification
Classes: energetic (party), peaceful (drive+calm), emotional (sad+romantic)
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🎮 GPU available: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(e)
else:
    print("💻 Using CPU (training will be slower)")


def create_advanced_cnn(input_shape=(224, 224, 3), num_classes=3):  # UPDATED: 3 classes
    """
    Advanced CNN for 3-class mood classification
    """
    model = models.Sequential([
        # Block 1: Initial feature extraction
        layers.Conv2D(64, (7, 7), strides=2, padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((3, 3), strides=2, padding='same'),
        layers.Dropout(0.2),
        
        # Block 2: Low-level features
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Block 3: Mid-level features
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Block 4: High-level features
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        # Dense classifier
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # UPDATED: 3 classes
    ])
    
    return model


def create_residual_cnn(input_shape=(224, 224, 3), num_classes=3):  # UPDATED: 3 classes
    """
    Residual CNN for 3-class mood classification
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial conv
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    # Residual block 1
    for _ in range(2):
        residual = x
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Residual block 2
    residual = layers.Conv2D(128, (1, 1), strides=1, padding='same')(x)
    for _ in range(2):
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)
        residual = x
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Residual block 3
    residual = layers.Conv2D(256, (1, 1), strides=1, padding='same')(x)
    for _ in range(2):
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)
        residual = x
    
    # Global pooling and classifier
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)  # UPDATED: 3 classes
    
    model = tf.keras.Model(inputs, outputs)
    return model


class DetailedProgressCallback(Callback):
    """Custom callback to show detailed progress"""
    def on_epoch_end(self, epoch, logs=None):
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   Train Loss: {logs['loss']:.4f} | Train Acc: {logs['accuracy']:.2%}")
        print(f"   Val Loss:   {logs['val_loss']:.4f} | Val Acc:   {logs['val_accuracy']:.2%}")
        if 'val_accuracy' in logs and logs['val_accuracy'] >= 0.7:
            print(f"   🎉 Excellent validation accuracy!")


def plot_training_history(history, save_path='../models/training_history.png'):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 Training history saved to: {save_path}")


def main():
    print("="*80)
    print("🎵 OPTIMIZED 3-CLASS MOOD CLASSIFICATION TRAINING PIPELINE")  # UPDATED
    print("="*80)
    
    # Create models directory
    models_dir = '../models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Check for spectrogram directories (priority: resized > enhanced > fusion)
    spectrogram_dirs = [
        "../mel_spectrograms_resized",
        "../mel_spectrograms_enhanced",
        "../mel_spectrograms_fusion"
    ]
    
    spectrogram_dir = None
    for dir_path in spectrogram_dirs:
        if os.path.exists(dir_path) and os.listdir(dir_path):
            spectrogram_dir = dir_path
            print(f"\n✅ Found spectrograms: {dir_path}")
            break
    
    if spectrogram_dir is None:
        print("\n❌ No spectrogram directory found!")
        print("Run one of these first:")
        print("  - python3 generate_optimized_spectrograms.py")
        print("  - python3 generate_final_hybrid_spectrograms.py")
        return
    
    # Auto-detect image size
    sample_mood = [d for d in os.listdir(spectrogram_dir) if os.path.isdir(os.path.join(spectrogram_dir, d))][0]
    sample_dir = os.path.join(spectrogram_dir, sample_mood)
    sample_file = os.path.join(sample_dir, os.listdir(sample_dir)[0])
    
    try:
        from PIL import Image
        with Image.open(sample_file) as img:
            img_width, img_height = img.size
            target_size = (img_width, img_height)
            print(f"📐 Detected image size: {img_width}x{img_height}")
    except:
        target_size = (224, 224)
        print(f"📐 Using default size: {target_size}")
    
    # Data augmentation - CRITICAL for small datasets
    print("\n🔄 Setting up data augmentation...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,           # Slight rotation
        width_shift_range=0.15,      # Horizontal shift
        height_shift_range=0.15,     # Vertical shift (time shift)
        shear_range=0.1,            # Shear transformation
        zoom_range=0.2,             # Zoom in/out
        horizontal_flip=True,        # Time reversal
        brightness_range=[0.7, 1.3], # Brightness variation
        fill_mode='nearest',
        validation_split=0.2
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Create generators
    batch_size = 16  # Adjust based on your GPU memory
    
    train_gen = train_datagen.flow_from_directory(
        spectrogram_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        color_mode='rgb',
        seed=42
    )
    
    val_gen = val_datagen.flow_from_directory(
        spectrogram_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        color_mode='rgb',
        seed=42
    )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training samples:   {train_gen.samples}")
    print(f"   Validation samples: {val_gen.samples}")
    print(f"   Classes: {list(train_gen.class_indices.keys())}")
    print(f"   Batch size: {batch_size}")
    
    # Class distribution
    print(f"\n📈 Class Distribution:")
    for class_name, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
        train_count = np.sum(train_gen.classes == idx)
        val_count = np.sum(val_gen.classes == idx)
        print(f"   {class_name:10s}: Train={train_count:3d}, Val={val_count:3d}")
    
    # Compute class weights (CRITICAL for imbalanced data)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"\n⚖️  Class Weights (handling imbalance):")
    for class_name, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
        print(f"   {class_name:10s}: {class_weight_dict[idx]:.3f}")
    
    # Model selection
    input_shape = (*target_size, 3)
    num_classes = len(train_gen.class_indices)
    
    print(f"\n🏗️  Building model...")
    print(f"   Input shape: {input_shape}")
    print(f"   Output classes: {num_classes}")
    
    # Choose model based on dataset size
    if train_gen.samples > 800:
        print("   Architecture: RESIDUAL CNN (larger dataset)")
        model = create_residual_cnn(input_shape, num_classes)
        initial_lr = 0.0005
    else:
        print("   Architecture: ADVANCED CNN (standard)")
        model = create_advanced_cnn(input_shape, num_classes)
        initial_lr = 0.001
    
    # Compile with label smoothing (reduces overconfidence)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=initial_lr,
            beta_1=0.9,
            beta_2=0.999
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    # Show model summary
    print("\n📋 Model Architecture:")
    model.summary()
    print(f"\n💾 Total parameters: {model.count_params():,}")
    
    # Setup callbacks
    checkpoint_path = os.path.join(models_dir, 'best_model.h5')
    final_model_path = os.path.join(models_dir, 'final_model.h5')
    log_path = os.path.join(models_dir, 'training_log.csv')
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=30,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
            min_lr=1e-7,
            cooldown=5,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(log_path, append=True),
        DetailedProgressCallback()
    ]
    
    # Training
    print("\n" + "="*80)
    print("🔥 STARTING 3-CLASS TRAINING")  # UPDATED
    print("="*80)
    print(f"\n💡 Strategy:")
    print(f"   • 3-class classification: energetic, peaceful, emotional")  # UPDATED
    print(f"   • Using class weights to handle imbalance")
    print(f"   • Heavy data augmentation for generalization")
    print(f"   • Label smoothing to prevent overconfidence")
    print(f"\n⏱️  Training will take 15-45 minutes depending on your hardware...")
    print()
    
    try:
        history = model.fit(
            train_gen,
            epochs=150,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=2
        )
        
        # Save final model
        model.save(final_model_path)
        
        # Plot training history
        plot_training_history(history)
        
        # Results summary
        print("\n" + "="*80)
        print("✅ 3-CLASS TRAINING COMPLETE!")  # UPDATED
        print("="*80)
        
        best_val_acc = max(history.history['val_accuracy'])
        best_val_loss = min(history.history['val_loss'])
        final_train_acc = history.history['accuracy'][-1]
        
        print(f"\n📊 Final Results:")
        print(f"   Best Validation Accuracy:  {best_val_acc:.2%}")
        print(f"   Best Validation Loss:      {best_val_loss:.4f}")
        print(f"   Final Training Accuracy:   {final_train_acc:.2%}")
        print(f"   Overfitting gap:           {abs(final_train_acc - best_val_acc):.2%}")
        
        # Performance assessment
        print(f"\n🎯 Performance Assessment:")
        if best_val_acc >= 0.75:  # UPDATED: Higher threshold for 3 classes
            print("   🌟 EXCELLENT! Model performing very well!")
        elif best_val_acc >= 0.65:
            print("   ✅ VERY GOOD! Solid performance for mood classification.")
        elif best_val_acc >= 0.55:
            print("   📊 GOOD! Reasonable performance, room for improvement.")
        else:
            print("   ⚠️  NEEDS IMPROVEMENT")
        
        # Recommendations based on results
        overfitting = final_train_acc - best_val_acc
        if overfitting > 0.15:
            print(f"\n💡 Model is overfitting (gap: {overfitting:.2%}). Try:")
            print("   • More data augmentation")
            print("   • Increase dropout rates")
            print("   • Add more training data")
        
        if best_val_acc < 0.60:  # UPDATED: Higher threshold for 3 classes
            print(f"\n💡 To improve accuracy, try:")
            print("   • Generate augmented dataset (hybrid approach)")
            print("   • Use transfer learning (VGG16/ResNet)")
            print("   • Increase model capacity")
            print("   • Train for more epochs")
        
        print(f"\n📁 Model saved to:")
        print(f"   Best:  {checkpoint_path}")
        print(f"   Final: {final_model_path}")
        print(f"   Log:   {log_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        model.save(os.path.join(models_dir, 'interrupted_model.h5'))
        print(f"💾 Model saved as 'interrupted_model.h5'")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()