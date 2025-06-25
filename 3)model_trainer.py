import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# --- Constants and Setup ---
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['jab', 'hook', 'uppercut'])
num_sequences = 50  # Must match the number of sequences collected
sequence_length = 20 # Must match the sequence length used for collection

# --- 1. Load the Data ---
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

print(f"Data loaded. Found {len(sequences)} sequences.")

# Convert to numpy arrays
X = np.array(sequences)
y = np.array(labels)

# The shape of X will be (num_sequences * num_actions, sequence_length, num_features)
# For ML model, we need to reshape it to (num_samples, num_features)
# So we flatten the sequence and frame dimensions
num_samples = X.shape[0]
X_reshaped = X.reshape(num_samples, -1)

print(f"Data reshaped to: {X_reshaped.shape}")

# --- 2. Split Data for Training and Testing ---
print("Splitting data into training and testing sets...")

# stratify=y ensures that the distribution of labels is the same in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# --- 3. Train the Machine Learning Model ---
print("Training the Random Forest model...")

# Initialize the model. You can tune these parameters.
# n_estimators: number of trees in the forest
# random_state: for reproducibility
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

print("Model training complete.")

# --- 4. Evaluate the Model ---
print("Evaluating model performance...")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

# --- 5. Save the Trained Model ---
print("Saving the trained model to 'boxing_model.pkl'...")

with open('boxing_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nProcess complete. Your trained model has been saved!")