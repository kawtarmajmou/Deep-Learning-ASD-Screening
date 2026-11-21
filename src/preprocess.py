import os
import glob
import numpy as np
import mne

# EEG filter range (important for ASD biomarkers)
LOW_FREQ = 1.0
HIGH_FREQ = 40.0


# Find .prm file (current + parent folders)
def find_prm_file(dat_path):
    current = os.path.dirname(dat_path)

    for _ in range(5):  # search up to 5 levels
        prm_files = glob.glob(os.path.join(current, "*.prm"))
        if prm_files:
            return prm_files[0]
        current = os.path.dirname(current)

    return None


# Extract metadata from .prm
def read_prm_file(prm_path):
    sfreq = None
    n_channels = None

    with open(prm_path, "r", errors="ignore") as f:
        for line in f:
            if "SamplingRate" in line:
                try:
                    sfreq = float(line.split()[-1])
                except:
                    pass

            if "SourceCh" in line or "ChannelCount" in line:
                try:
                    n_channels = int(line.split()[-1])
                except:
                    pass

    return sfreq, n_channels


# Load .dat file manually
def load_bci2000_dat(dat_path):
    prm_path = find_prm_file(dat_path)

    # Default fallback (used if no .prm)
    sfreq = 256
    n_channels = 16

    if prm_path:
        print(f"   [FOUND] PRM: {os.path.basename(prm_path)}")

        sfreq_read, ch_read = read_prm_file(prm_path)

        if sfreq_read:
            sfreq = sfreq_read
        if ch_read:
            n_channels = ch_read
    else:
        print("   [WARNING] No .prm found - using default values")

    # Read binary data
    data = np.fromfile(dat_path, dtype=np.int16)

    # Safe reshape
    total_samples = len(data) // n_channels
    data = data[:total_samples * n_channels]
    data = data.reshape((total_samples, n_channels)).T

    # Channel names
    ch_names = [f"EEG{i}" for i in range(n_channels)]

    # Create MNE Raw object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    return raw


# Main pipeline
def process_all_subjects():
    data_root = "data"
    output_root = "processed"

    os.makedirs(output_root, exist_ok=True)

    dat_files = glob.glob(os.path.join(data_root, "**/*.dat"), recursive=True)

    if not dat_files:
        print("[ERROR] No .dat files found")
        return

    print(f"[INFO] Found {len(dat_files)} .dat files\n")

    for dat_path in dat_files:
        try:
            subject_name = os.path.basename(dat_path).replace(".dat", "")
            print(f"[PROCESSING] {subject_name}")

            raw = load_bci2000_dat(dat_path)

            # Band-pass filter (ASD EEG range)
            raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ, verbose=False)

            # Re-reference (important for EEG quality)
            raw.set_eeg_reference('average', verbose=False)

            # Save (FIXED naming)
            output_path = os.path.join(output_root, f"{subject_name}_raw.fif")
            raw.save(output_path, overwrite=True, verbose=False)

            print(f"[SUCCESS] {subject_name}\n")

        except Exception as e:
            print(f"[ERROR] {dat_path}")
            print(f"   Details: {str(e)}\n")


# Run script
if __name__ == "__main__":
    process_all_subjects()