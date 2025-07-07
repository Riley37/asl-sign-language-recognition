import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def create_simple_asl_model(input_shape, num_classes):
    """Simple CNN to verify basic learning works."""
    model = keras.Sequential([
        # Input normalization
        layers.Rescaling(1./255, input_shape=input_shape),
        
        # Simple feature extraction
        layers.Conv2D(16, (5, 5), activation='relu', padding='same'),
        layers.MaxPooling2D((4, 4)),  # 200→50
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((5, 5)),  # 50→10
        
        # Simple classifier
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def load_asl_data(data_dir, img_size=(200, 200), batch_size=32):
    train_data_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {train_data_dir}")
    
    letter_classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    available_classes = sorted([d for d in os.listdir(train_data_dir) 
                               if os.path.isdir(os.path.join(train_data_dir, d)) and d in letter_classes])
    
    train_ds = keras.utils.image_dataset_from_directory(
        train_data_dir,
        image_size=img_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=123,
        label_mode='categorical',
        class_names=available_classes
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        train_data_dir,
        image_size=img_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=123,
        label_mode='categorical',
        class_names=available_classes
    )
    
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return True
        except RuntimeError:
            return False
    return False


class StopAtTargetAccuracy(keras.callbacks.Callback):
    """Stop training when reaching target validation accuracy to prevent overfitting."""
    def __init__(self, target_accuracy=0.95):
        super().__init__()
        self.target_accuracy = target_accuracy
        
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy', 0)
        if val_acc >= self.target_accuracy:
            print(f"Reached target validation accuracy of {val_acc:.1%}")
            self.model.stop_training = True

def main():
    has_gpu = configure_gpu()
    
    DATA_DIR = 'data'
    MODEL_DIR = 'models'
    IMG_SIZE = (200, 200)
    BATCH_SIZE = 8 if has_gpu else 4
    EPOCHS = 20
    LEARNING_RATE = 0.0001
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    train_ds, val_ds = load_asl_data(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    num_classes = len(train_ds.class_names)
    
    model = create_simple_asl_model(IMG_SIZE + (3,), num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    callbacks = [
        StopAtTargetAccuracy(target_accuracy=0.95),  # Stop at 95% to prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, 'best_asl_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    model.save(os.path.join(MODEL_DIR, 'final_asl_model.h5'))
    
    with open(os.path.join(MODEL_DIR, 'asl_class_names.txt'), 'w') as f:
        for class_name in train_ds.class_names:
            f.write(f"{class_name}\n")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'asl_training_history.png'))
    plt.show()

if __name__ == "__main__":
    main()