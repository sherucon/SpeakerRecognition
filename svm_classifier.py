import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ================= CONFIGURATION =================
INPUT_FILE = "svd_features.npz"

def main():
    print("--- Stage 3: SVM Classification (SMO) ---")

    # 1. Load the data from Stage 2
    if not os.path.exists(INPUT_FILE):
        print(f"Error: '{INPUT_FILE}' not found. Run script 2 first.")
        return

    data = np.load(INPUT_FILE)
    X = data['X']
    y = data['y']
    print(f"Loaded SVD features: {X.shape}")

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 3. Initialize SVM 
    # kernel='linear' is used as per the paper's specific comparison
    # scikit-learn's SVC uses libsvm, which implements SMO
    clf = SVC(kernel='linear', C=1.0)

    # 4. Train
    print("Training SVM model...")
    clf.fit(X_train, y_train)

    # 5. Predict & Evaluate
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nFinal Accuracy: {acc * 100:.2f}%")
    print("-" * 30)
    print("Detailed Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()