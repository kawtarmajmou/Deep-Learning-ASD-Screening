import os

# --- PATH SETTINGS ---
# This points to the folder containing your 31 subject folders
RAW_DATA_PATH = os.path.join("data")

# This is where your processed spectrogram images will be saved
PROCESSED_DATA_PATH = os.path.join("data", "spectrograms")

# Create the folder if it doesn't exist yet
if not os.path.exists(PROCESSED_DATA_PATH):
    os.makedirs(PROCESSED_DATA_PATH)

# --- EEG PREPROCESSING SETTINGS ---
# These values are standard for ASD biomarker extraction
LOW_FREQ = 1.0   # Removes DC offset and slow drifts
HIGH_FREQ = 40.0 # Removes muscle noise and power line interference

# --- MODEL SETTINGS ---
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001