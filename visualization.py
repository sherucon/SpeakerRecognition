import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

# ================= CONFIGURATION =================
INPUT_RAW = "raw_features.npz"   # From Stage 1 (for SVD plot)
INPUT_SVD = "svd_features.npz"   # From Stage 2 (for SVM plot)

def plot_confusion_matrix(y_test, y_pred, labels):
    """
    Generates a heatmap showing correct vs incorrect predictions.
    """
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Speaker')
    plt.ylabel('True Speaker')
    plt.title('Confusion Matrix: Speaker Identification')
    plt.tight_layout()
    plt.show()

def plot_svm_boundaries(X, y):
    """
    Visualizes the Linear SVM Decision Boundaries in 2D.
    Note: This trains a NEW 2D model just for visualization purposes,
    using the top 2 SVD components.
    """
    print("Generating 2D Decision Boundary Plot...")
    
    # We only use the first 2 dimensions for the plot
    X_2d = X[:, :2]
    
    # Encode labels to integers for plotting
    unique_labels = np.unique(y)
    y_int = np.zeros_like(y, dtype=int)
    for i, label in enumerate(unique_labels):
        y_int[y == label] = i

    # Train a 2D SVM just for this plot
    clf_2d = SVC(kernel='linear', C=1.0)
    clf_2d.fit(X_2d, y_int)

    # Create a mesh to plot
    h = .02  # step size in the mesh
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict for each point in the mesh
    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    
    # Plot the training points
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_int, cmap=plt.cm.coolwarm, edgecolors='k', s=80)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(unique_labels), title="Speakers")
    plt.xlabel('SVD Component 1')
    plt.ylabel('SVD Component 2')
    plt.title('SVM Decision Boundaries (2D Projection)')
    plt.show()

def plot_svd_variance(X_raw):
    """
    Plots the 'Scree Plot' showing how much information is retained
    by each SVD component.
    """
    print("Generating SVD Variance Plot...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Compute full SVD to see all components
    n_features = X_scaled.shape[1]
    svd = TruncatedSVD(n_components=n_features-1, random_state=42)
    svd.fit(X_scaled)
    
    explained_variance = svd.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(1, len(cumulative_variance)+1), cumulative_variance, where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Components (SVD Dimensions)')
    plt.title('SVD Feature Compression Analysis')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def main():
    # 1. Load Data
    if not os.path.exists(INPUT_SVD) or not os.path.exists(INPUT_RAW):
        print("Error: Data files not found. Run scripts 1 and 2 first.")
        return

    data_svd = np.load(INPUT_SVD)
    X_svd = data_svd['X']
    y = data_svd['y']
    
    data_raw = np.load(INPUT_RAW)
    X_raw = data_raw['X']

    # 2. Train a model to get predictions for the Confusion Matrix
    X_train, X_test, y_train, y_test = train_test_split(X_svd, y, test_size=0.2, random_state=42)
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 3. Generate Plots
    # A. Confusion Matrix
    print("Plotting Confusion Matrix...")
    unique_labels = sorted(list(set(y)))
    plot_confusion_matrix(y_test, y_pred, unique_labels)

    # B. SVM Decision Boundaries (Visualization)
    # We use the full dataset here to see how all speakers are separated
    plot_svm_boundaries(X_svd, y)

    # C. SVD Variance
    plot_svd_variance(X_raw)

if __name__ == "__main__":
    main()