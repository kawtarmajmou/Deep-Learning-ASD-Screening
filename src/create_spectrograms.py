import os
import mne
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Settings
PROCESSED_ROOT = "processed"
SPECTROGRAM_ROOT = os.path.join("data", "spectrograms")
SEGMENT_DURATION = 2  # Each image represents 2 seconds of brain activity

def create_spectrograms_from_fif():
    # 1. Prepare folder structure for CNN classification
    for label in ["Autism", "Normal"]:
        os.makedirs(os.path.join(SPECTROGRAM_ROOT, label), exist_ok=True)

    # 2. Find the .fif files created by your preprocess.py
    fif_files = glob.glob(os.path.join(PROCESSED_ROOT, "*_raw.fif"))

    if not fif_files:
        print("Error: No .fif files found in 'processed/'. Run preprocess.py first!")
        return

    print(f"Found {len(fif_files)} preprocessed files. Generating spectrograms...")

    for fif_path in fif_files:
        file_name = os.path.basename(fif_path)
        
        # Sort into Autism or Normal folders based on the filename
        label = "Autism" if "Autism" in file_name else "Normal"
        
        try:
            # Load the pre-filtered data
            raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            
            # 3. Calculate samples for 2-second windows
            samples_per_seg = int(SEGMENT_DURATION * sfreq)
            num_segments = data.shape[1] // samples_per_seg
            
            for i in range(num_segments):
                start = i * samples_per_seg
                end = start + samples_per_seg
                
                # We average all 16 channels to create a unified 'brain state' image
                segment_data = np.mean(data[:, start:end], axis=0)
                
                # 4. Compute the Spectrogram (Frequency vs. Time)
                f, t, Sxx = signal.spectrogram(segment_data, sfreq)
                
                # Create the image
                plt.figure(figsize=(3, 3)) # 3x3 inches is ideal for CNN input
                # 10 * np.log10 converts the power to Decibels (dB) for better contrast
                plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='jet')
                plt.axis('off') # Remove axes so the CNN only sees data
                
                # Save the image
                subject_id = file_name.replace("_raw.fif", "")
                save_path = os.path.join(SPECTROGRAM_ROOT, label, f"{subject_id}_seg_{i}.png")
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()

            print(f"Finished: {file_name} -> Created {num_segments} images")

        except Exception as e:
            print(f"Failed to process {file_name}: {e}")

if __name__ == "__main__":
    create_spectrograms_from_fif()