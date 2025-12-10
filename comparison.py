import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

# ================= CONFIGURATION =================
INPUT_FILE = "svd_features.npz"
NUM_SPEAKERS = 8  # We know there are 8 speakers in the dataset

def load_data():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"'{INPUT_FILE}' not found. Please run the SVD script first.")
    data = np.load(INPUT_FILE)
    return data['X'], data['y']

def train_logistic_regression(X_train, X_test, y_train, y_test):
    print("\n" + "="*40)
    print("LOGISTIC REGRESSION")
    print("="*40)
    
    clf = LogisticRegression(random_state=42, max_iter=1000) 
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))
    
    return acc

def train_svm(X_train, X_test, y_train, y_test):
    print("\n" + "="*40)
    print("SVM (LINEAR)")
    print("="*40)
    
    clf = SVC(kernel='linear', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))
    
    return acc

def train_kmeans_classifier(X_train, X_test, y_train, y_test):
    print("\n" + "="*40)
    print("K-MEANS CLUSTERING (VQ)")
    print("="*40)
    
    # 1. Fit K-Means
    kmeans = KMeans(n_clusters=NUM_SPEAKERS, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_train)
    
    # 2. Map Clusters to Labels
    cluster_map = {}
    for i in range(NUM_SPEAKERS):
        mask = (clusters == i)
        if np.sum(mask) == 0:
            cluster_map[i] = y_train[0] 
            continue
            
        true_labels_in_cluster = y_train[mask]
        
        # Use np.unique to handle both string and int labels safely
        values, counts = np.unique(true_labels_in_cluster, return_counts=True)
        most_common = values[np.argmax(counts)]
        cluster_map[i] = most_common
        
    # 3. Predict on Test Set
    test_clusters = kmeans.predict(X_test)
    y_pred = np.array([cluster_map[c] for c in test_clusters])
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nDetailed Report:")
    # We explicitly provide labels to ensure the report includes all speakers
    unique_labels = sorted(list(set(y_test)))
    print(classification_report(y_test, y_pred, labels=unique_labels))
    
    return acc

def plot_comparison(results):
    models = list(results.keys())
    accuracies = list(results.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, 
                 f"{yval*100:.1f}%", ha='center', va='bottom', fontweight='bold')
        
    plt.ylim(0, 1.1)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison (SVD Features)")
    plt.grid(axis='y', alpha=0.3)
    plt.show()

def main():
    try:
        X, y = load_data()
        
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        
        results = {}
        
        # Run all three models
        results['Logistic Regression'] = train_logistic_regression(X_train, X_test, y_train, y_test)
        results['K-Means'] = train_kmeans_classifier(X_train, X_test, y_train, y_test)
        results['SVM (Linear)'] = train_svm(X_train, X_test, y_train, y_test)
        
        # Plot
        plot_comparison(results)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()