import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from collections import Counter

# Load dataset
CSV_FILE = "hand_gestures.csv"  # Update with correct path
df = pd.read_csv(CSV_FILE)

# Separate features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Save label encoder
np.save("models/label_encoder.npy", label_encoder)

# ✅ Check Label Distribution
label_counts = Counter(y)
print("Label Distribution Before Balancing:", label_counts)


# ✅ Data Balancing: Augment Rare Classes & Downsample Frequent Classes
def balance_dataset(X, y, min_samples=4000, max_samples=6000):
    """Balances dataset by augmenting rare classes and downsampling frequent classes."""
    X_balanced, y_balanced = [], []
    class_counts = Counter(y)

    for label in np.unique(y):
        indices = np.where(y == label)[0]
        num_samples = len(indices)

        if num_samples < min_samples:
            # Augment rare classes
            num_to_add = min_samples - num_samples
            sampled_indices = np.random.choice(indices, num_to_add, replace=True)
            X_balanced.extend(X[sampled_indices])
            y_balanced.extend(y[sampled_indices])

        elif num_samples > max_samples:
            # Downsample frequent classes
            sampled_indices = np.random.choice(indices, max_samples, replace=False)
            X_balanced.extend(X[sampled_indices])
            y_balanced.extend(y[sampled_indices])
        else:
            # Keep original
            X_balanced.extend(X[indices])
            y_balanced.extend(y[indices])

    return np.array(X_balanced), np.array(y_balanced)


X_balanced, y_balanced = balance_dataset(X, y)

# ✅ New Label Distribution After Balancing
label_counts_balanced = Counter(y_balanced)
print("Label Distribution After Balancing:", label_counts_balanced)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42,
                                                    stratify=y_balanced)


# ✅ Augmentation with Noise, Rotation, and Scaling
def augment_data(X, noise_factor=0.02, rotation_range=15, scale_range=0.1):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=X.shape)
    X_noisy = X + noise
    scaling_factor = np.random.uniform(1 - scale_range, 1 + scale_range, size=(X.shape[0], 1))
    return X_noisy * scaling_factor


X_train_augmented = augment_data(X_train)

# ✅ Class Weights Calculation
class_weights = {i: 1.0 / count for i, count in label_counts_balanced.items()}
print("Class Weights Applied:", class_weights)

# ✅ Model Architecture Improvement
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# ✅ Hyperparameter Tuning
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ✅ Early Stopping to Prevent Overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ✅ Train Model with Class Weights
model.fit(X_train_augmented, y_train, validation_data=(X_test, y_test),
          epochs=100, batch_size=32, callbacks=[early_stopping], class_weight=class_weights)

# ✅ Save Model
model.save("models/sign_model.keras")
print("✅ Model saved successfully!")
