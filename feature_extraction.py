import os
import numpy as np
import librosa

# ================= CONFIGURATION =================
DATASET_PATH = "./data/"   # Root directory containing folders 1, 2, 3...
OUTPUT_FILE = "raw_features.npz"
SAMPLE_RATE = 48000  # Standard sample rate for audio
# MFCC parameters   
FRAME_LENGTH_MS = 0.03
N_MFCC = 20
N_FFT = int(SAMPLE_RATE * FRAME_LENGTH_MS) 

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Exact technique: Hamming window, 20 coefficients
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=N_MFCC,
            n_fft=N_FFT, 
            window='hamming'
        )

        # Exact technique: Exclude first component (c0)
        mfccs = mfccs[1:, :] 
        
        # Calculate mean to create a single vector per speaker sample
        mfcc_mean = np.mean(mfccs.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    features = []
    labels = []
    speaker_dirs = [str(i) for i in range(1, 9)] # Folders 1 through 8
    
    print("--- Stage 1: Feature Extraction (MFCC) ---")
    
    for speaker_id in speaker_dirs:
        dir_path = os.path.join(DATASET_PATH, speaker_id)
        if not os.path.exists(dir_path): continue
            
        print(f"Processing Speaker folder: {speaker_id}")
        for filename in os.listdir(dir_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(dir_path, filename)
                data = extract_features(file_path)
                if data is not None:
                    features.append(data)
                    labels.append(speaker_id)

    # Save to disk for the next file
    np.savez(OUTPUT_FILE, X=np.array(features), y=np.array(labels))
    print(f"\nSuccess! Extracted {len(features)} samples.")
    print(f"Data saved to '{OUTPUT_FILE}'. You can now run the next script.")

if __name__ == "__main__":
    main()