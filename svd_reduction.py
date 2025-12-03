import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

# ================= CONFIGURATION =================
INPUT_FILE = "raw_features.npz"
OUTPUT_FILE = "svd_features.npz"
N_COMPONENTS = 10  # Reducing from 19 features down to 10

def main():
    print("--- Stage 2: Dimensionality Reduction (SVD) ---")
    
    # 1. Load the data from Stage 1
    if not os.path.exists(INPUT_FILE):
        print(f"Error: '{INPUT_FILE}' not found. Run script 1 first.")
        return

    data = np.load(INPUT_FILE)
    X = data['X']
    y = data['y']
    print(f"Loaded raw feature shape: {X.shape}")

    # 2. Standardize (Scale) - Crucial before SVD
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Apply SVD
    print(f"Applying SVD to reduce dimensions to {N_COMPONENTS}...")
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    X_svd = svd.fit_transform(X_scaled)
    
    print(f"New feature shape: {X_svd.shape}")

    # 4. Save the reduced data for Stage 3
    np.savez(OUTPUT_FILE, X=X_svd, y=y)
    print(f"Reduced data saved to '{OUTPUT_FILE}'. You can now run the next script.")

# Helper to ensure 'os' is imported if you copy-paste just this block
import os
if __name__ == "__main__":
    main()