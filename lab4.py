import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse


def compute_euclidean_distance(point1, point2):
    """
    Return Euclidean distance between two N-dimensional points.
    """
    return np.linalg.norm(point1 - point2)

def render_class_comparison(data_a, data_b, center_a, center_b):
    """
    Plot PCA-based 2D view of two classes including:
    - Individual data points
    - Class centroids
    - Spread representation using ellipses
    - Line joining centroids to show class separation
    """
    # Merge for PCA transformation
    merged_data = np.vstack((data_a, data_b))
    pca_model = PCA(n_components=2)
    pca_data = pca_model.fit_transform(merged_data)

    # Separate transformed class data
    proj_a = pca_data[:len(data_a)]
    proj_b = pca_data[len(data_a):]

    # Project the class centroids to 2D
    pca_centroids = pca_model.transform([center_a, center_b])
    proj_center_a, proj_center_b = pca_centroids

    # Plotting data points
    plt.figure(figsize=(8, 6))
    plt.scatter(proj_a[:, 0], proj_a[:, 1], color='skyblue', label='Group A')
    plt.scatter(proj_b[:, 0], proj_b[:, 1], color='salmon', label='Group B')

    # Plotting centroids
    plt.scatter(*proj_center_a, color='blue', s=120, marker='X', label='Centroid A')
    plt.scatter(*proj_center_b, color='darkred', s=120, marker='X', label='Centroid B')

    # Draw line between centroids
    plt.plot([proj_center_a[0], proj_center_b[0]],
             [proj_center_a[1], proj_center_b[1]],
             color='black', linestyle='--', linewidth=1.8, label='Centroid Distance')

    # Helper function for drawing ellipse around class spread
    def plot_ellipse(center, data_pts, clr):
        covariance = np.cov(data_pts, rowvar=False)
        eigen_vals, eigen_vecs = np.linalg.eigh(covariance)
        order = eigen_vals.argsort()[::-1]
        eigen_vals, eigen_vecs = eigen_vals[order], eigen_vecs[:, order]
        theta = np.degrees(np.arctan2(*eigen_vecs[:, 0][::-1]))
        w, h = 2 * np.sqrt(eigen_vals)
        ellipse = Ellipse(xy=center, width=w, height=h, angle=theta,
                          edgecolor=clr, fc='none', lw=2, linestyle='--')
        plt.gca().add_patch(ellipse)

    plot_ellipse(np.mean(proj_a, axis=0), proj_a, 'skyblue')
    plot_ellipse(np.mean(proj_b, axis=0), proj_b, 'salmon')

    # Final plot setup
    plt.title("PCA Projection of Class Distributions")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # File location for input data
    csv_path = '/Users/keert/Downloads/features_lab3_labeled (1).csv'
    data = pd.read_csv(csv_path)

    # Feature columns to be analyzed
    selected_columns = ['mfcc1', 'rms', 'zcr', 'pitch_std', 'silence_pct']

    # Get unique classes and choose first two
    class_labels = data['class'].unique()
    label1, label2 = class_labels[0], class_labels[1]

    # Compute stats for both classes
    mean_vec1, std_vec1, data1 = extract_class_features(data, selected_columns, label1)
    mean_vec2, std_vec2, data2 = extract_class_features(data, selected_columns, label2)

    # Compute inter-class centroid gap
    centroid_gap = compute_euclidean_distance(mean_vec1, mean_vec2)

    # Output results
    print(f"--- Mean Vector for Class {label1} ---\n{pd.Series(mean_vec1, index=selected_columns)}\n")
    print(f"--- Standard Deviation for Class {label1} ---\n{pd.Series(std_vec1, index=selected_columns)}\n")
    print(f"--- Mean Vector for Class {label2} ---\n{pd.Series(mean_vec2, index=selected_columns)}\n")
    print(f"--- Standard Deviation for Class {label2} ---\n{pd.Series(std_vec2, index=selected_columns)}\n")
    print(f"=== Euclidean Distance Between Class {label1} and Class {label2}: {centroid_gap:.4f}")

    # Visualize PCA-based comparison
    render_class_comparison(data1, data2, mean_vec1, mean_vec2)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def feature_statistics_and_histogram(df, column_name, num_bins=10):
    """
    Generate histogram for a specific feature and calculate its mean and variance.

    Parameters:
        df (pd.DataFrame): Input data
        column_name (str): Feature column to visualize
        num_bins (int): Number of bins for the histogram plot

    Returns:
        tuple: (mean_result, variance_result)
    """
    # Get the data values from the selected column
    values = df[column_name].values

    # Calculate basic statistics
    avg = np.mean(values)
    variance = np.var(values)

    # Plot the histogram with a vertical line for mean
    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=num_bins, color='lightgreen', edgecolor='black')
    plt.axvline(avg, color='red', linestyle='--', linewidth=2, label=f'Avg = {avg:.2f}')

    # Plot decorations
    plt.title(f'Distribution of {column_name}')
    plt.xlabel(f'{column_name}')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return avg, variance


if __name__ == "__main__":
    # File path to the input CSV file
    dataset_path = '/Users/keert/Downloads/features_lab3_labeled (1).csv'
    
    # Read CSV into a DataFrame
    data_frame = pd.read_csv(dataset_path)

    # Select a feature column for visualization
    selected_feature = 'pitch_std'

    # Perform analysis and visualization
    avg_val, var_val = feature_statistics_and_histogram(data_frame, selected_feature)

    # Display results
    print(f"\n Feature Analysis: {selected_feature}")
    print(f"Mean: {avg_val:.4f}")
    print(f"Variance: {var_val:.4f}")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def minkowski_dist(vec_a, vec_b, order):
    """
    Compute Minkowski distance between two vectors using a specified order (r).
    
    Parameters:
        vec_a (array): First vector
        vec_b (array): Second vector
        order (int): Minkowski distance order (r)
    
    Returns:
        float: Computed distance
    """
    return np.power(np.sum(np.abs(vec_a - vec_b) ** order), 1 / order)

def evaluate_minkowski_series(dataset, idx_a, idx_b, r_values):
    """
    Evaluate Minkowski distances for a pair of vectors across multiple r values.
    
    Parameters:
        dataset (pd.DataFrame): Data containing feature vectors
        idx_a (int): Index of first vector
        idx_b (int): Index of second vector
        r_values (list): List of r values
    
    Returns:
        list: Distance values corresponding to each r
    """
    vector_a = dataset.iloc[idx_a].values
    vector_b = dataset.iloc[idx_b].values
    result_distances = []

    for r in r_values:
        result = minkowski_dist(vector_a, vector_b, r)
        result_distances.append(result)
        print(f"Distance with r={r}: {result:.4f}")

    return result_distances

def plot_distance_variation(r_list, dist_list, idx1, idx2):
    """
    Create a line plot showing how Minkowski distance changes with r.
    
    Parameters:
        r_list (list): Range of r values
        dist_list (list): Corresponding distances
        idx1 (int): First vector index
        idx2 (int): Second vector index
    """
    plt.figure(figsize=(8, 6))
    plt.plot(r_list, dist_list, color='darkgreen', marker='o', linewidth=2)
    plt.title(f'Minkowski Distance vs r (Vector {idx1+1} vs Vector {idx2+1})')
    plt.xlabel('Order r')
    plt.ylabel('Minkowski Distance')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # File path for the CSV data
    csv_file = '/Users/keert/Downloads/features_lab3_labeled (1).csv'
    
    # Read the dataset
    df_data = pd.read_csv(csv_file)

    # Feature columns to include (excluding 'filename' and 'class')
    selected_features = ['mfcc1', 'rms', 'zcr', 'pitch_std', 'silence_pct']
    feature_vectors = df_data[selected_features]

    # Indices of the vectors to compare
    vec_index1 = 0
    vec_index2 = 1

    # r values from 1 to 10
    r_range = list(range(1, 11))

    # Calculate distances across r values
    distances_result = evaluate_minkowski_series(feature_vectors, vec_index1, vec_index2, r_range)

    # Visualize the distances
    plot_distance_variation(r_range, distances_result, vec_index1, vec_index2)
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Specify the path to the dataset
    data_path = '/Users/keert/Downloads/features_lab3_labeled (1).csv'
    # Load the dataset into a DataFrame
    dataset = pd.read_csv(data_path)

    # Define the feature columns and label column
    input_features = ['mfcc1', 'rms', 'zcr', 'pitch_std', 'silence_pct']
    X_data = dataset[input_features]      # Features
    y_label = dataset['class']            # Target label

    # Split data into training and test sets (70% train, 30% test)
    X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(
        X_data, y_label, test_size=0.3, random_state=42
    )

    # Display the number of samples in each split
    print(f"Total data points: {len(dataset)}")
    print(f"Training set size: {len(X_train_set)}")
    print(f"Testing set size: {len(X_test_set)}\n")

    # Display a few sample labels from training and testing sets
    print(">>> First 5 labels in the training set:")
    print(y_train_set.head(), "\n")

    print(">>> First 5 labels in the testing set:")
    print(y_test_set.head())
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

print("===== k-Nearest Neighbors Classification =====")

# Define the path to the dataset file
csv_file = '/Users/keert/Downloads/features_lab3_labeled (1).csv'

# Load dataset into a DataFrame
data_frame = pd.read_csv(csv_file)

# Define input attributes and target labels
input_columns = ['mfcc1', 'rms', 'zcr', 'pitch_std', 'silence_pct']
features = data_frame[input_columns]
labels = data_frame['class']

# Split the dataset into training and testing subsets
X_train_data, X_test_data, y_train_labels, y_test_labels = train_test_split(
    features, labels, test_size=0.3, random_state=42
)

# Create and train the kNN model with k=3 neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train_data, y_train_labels)

# Perform predictions on the test set
predicted_labels = knn_classifier.predict(X_test_data)

# Evaluate prediction performance
test_accuracy = accuracy_score(y_test_labels, predicted_labels)
print(f"Test Set Accuracy: {test_accuracy * 100:.2f}%\n")

# Show detailed classification results
print("Detailed Classification Report:")
print(classification_report(y_test_labels, predicted_labels, digits=2))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

# Generate 1000 random values from standard normal distribution
random_values = np.random.randn(1000)

# Part A: Histogram plot
plt.figure(figsize=(7, 5))
plt.hist(random_values, bins=30, color='lightgreen', edgecolor='black')
plt.title("Normal Distribution - Histogram")
plt.xlabel("Data Values")
plt.ylabel("Count")
plt.grid(True)
plt.show()

# Part B: KDE vs Actual PDF
plt.figure(figsize=(7, 5))

# KDE using seaborn
sns.kdeplot(random_values, color='darkred', label='KDE Curve', linewidth=2)

# Actual Normal Distribution PDF
x_range = np.linspace(-5, 5, 1000)
pdf_values = norm.pdf(x_range, loc=0, scale=1)
plt.plot(x_range, pdf_values, 'b--', label='True Normal PDF', linewidth=2)

plt.title("PDF Comparison: KDE vs Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc

# Load CSV data
csv_path = '/Users/keert/Downloads/features_lab3_labeled (1).csv'
dataset = pd.read_csv(csv_path)

# Keep only numeric columns
numeric_data = dataset.select_dtypes(include=[np.number])

# Keep rows where the class is 1 or 2 for binary classification
binary_data = numeric_data[numeric_data.iloc[:, -1].isin([1, 2])]

# Separate features and targets
features = binary_data.iloc[:, :-1].values
labels = binary_data.iloc[:, -1].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Fit kNN classifier
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Predict probability scores
probabilities = knn_model.predict_proba(X_test)

# Convert class labels to binary
binarizer = LabelBinarizer()
binary_labels = binarizer.fit_transform(y_test).ravel()

# ROC curve based on probabilities for class 2
fpr, tpr, _ = roc_curve(binary_labels, probabilities[:, 1])
roc_score = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC Curve (AUC = {roc_score:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Baseline (AUC = 0.5)')
plt.title('ROC Curve for Binary kNN Classification (1 vs 2)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset from CSV
csv_path = '/Users/keert/Downloads/features_lab3_labeled (1).csv'
df_raw = pd.read_csv(csv_path)

# Keep numeric columns only
data_numeric = df_raw.select_dtypes(include=[np.number])

# Features and labels split
features = data_numeric.iloc[:, :-1].values
targets = data_numeric.iloc[:, -1].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=42)

# Define different distance metric configurations
distance_configs = [
    ("euclidean", {}),
    ("manhattan", {}),
    ("chebyshev", {}),
    ("minkowski", {"p": 3}),
    ("minkowski", {"p": 4})
]

acc_results = []
metric_names = []

print("\n=== kNN Accuracy Results with Various Distance Metrics ===")
for metric, params in distance_configs:
    model = KNeighborsClassifier(n_neighbors=3, metric=metric, **params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    label = f"{metric} {params}" if params else f"{metric} ()"
    print(f"Distance: {label} --> Accuracy: {accuracy:.4f}")
    metric_names.append(label)
    acc_results.append(accuracy)

# Visualize the accuracy comparison
plt.figure(figsize=(10, 6))
plt.bar(metric_names, acc_results, color='steelblue')
plt.title("kNN Accuracy with Different Distance Metrics")
plt.xlabel("Distance Metric")
plt.ylabel("Accuracy Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=30)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset from CSV
csv_path = '/Users/keert/Downloads/features_lab3_labeled (1).csv'
df_raw = pd.read_csv(csv_path)

# Keep numeric columns only
data_numeric = df_raw.select_dtypes(include=[np.number])

# Features and labels split
features = data_numeric.iloc[:, :-1].values
targets = data_numeric.iloc[:, -1].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=42)

# Define different distance metric configurations
distance_configs = [
    ("euclidean", {}),
    ("manhattan", {}),
    ("chebyshev", {}),
    ("minkowski", {"p": 3}),
    ("minkowski", {"p": 4})
]

acc_results = []
metric_names = []

print("\n=== kNN Accuracy Results with Various Distance Metrics ===")
for metric, params in distance_configs:
    model = KNeighborsClassifier(n_neighbors=3, metric=metric, **params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    label = f"{metric} {params}" if params else f"{metric} ()"
    print(f"Distance: {label} --> Accuracy: {accuracy:.4f}")
    metric_names.append(label)
    acc_results.append(accuracy)

# Visualize the accuracy comparison
plt.figure(figsize=(10, 6))
plt.bar(metric_names, acc_results, color='steelblue')
plt.title("kNN Accuracy with Different Distance Metrics")
plt.xlabel("Distance Metric")
plt.ylabel("Accuracy Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=30)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
