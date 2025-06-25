import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.model_selection import train_test_split

# --- 1. Constants and Data Loading ---
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['jab', 'hook', 'uppercut'])
num_sequences = 50  # Must match the number of sequences collected
sequence_length = 20 # Must match the sequence length used for collection

print("Loading data...")
sequences, labels = [], []
label_map = {label: num for num, label in enumerate(actions)}

for action in actions:
    for sequence in range(num_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = np.array(labels)

print(f"Data shape: {X.shape}") # Should be (num_samples, sequence_length, num_features)

# --- 2. Splitting Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# --- 3. Building the LSTM Model ---
print("Building the LSTM model...")
model = Sequential([
    # Input layer specifies the shape of each sample
    Input(shape=(sequence_length, X.shape[2])), # (20, num_features)
    
    # First LSTM layer with 64 units. return_sequences=True because we are stacking another LSTM layer
    LSTM(64, return_sequences=True, activation='relu'),
    
    # Second LSTM layer with 128 units. return_sequences=False as it's the last LSTM layer
    LSTM(128, return_sequences=False, activation='relu'),
    
    # A standard fully connected layer
    Dense(64, activation='relu'),
    Dropout(0.5), # Dropout helps prevent overfitting
    
    # Output layer with 'softmax' activation for multi-class classification
    Dense(actions.shape[0], activation='softmax')
])

# --- 4. Compiling the Model ---
# Using SparseCategoricalCrossentropy because our labels (y) are integers (0, 1, 2)
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the model architecture
model.summary()

# --- 5. Training the Model ---
print("\nTraining the model...")
# An epoch is one full pass through the entire training dataset
# We also use a portion of the training data as a validation set to monitor for overfitting
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# --- 6. Evaluating the Model ---
print("\nEvaluating model performance on the test set...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")


# --- 7. Saving the Trained Model ---
MODEL_NAME = 'action_model.keras'
print(f"\nSaving the trained model to '{MODEL_NAME}'...")
model.save(MODEL_NAME)
print("Model saved successfully.")