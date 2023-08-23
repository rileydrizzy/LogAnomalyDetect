## Heruristic Benchmark 
# TODO implement HB pr Dummy Classifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Generate synthetic imbalanced data (replace this with your actual data)
np.random.seed(42)
minority_class_size = 100
majority_class_size = 1000
minority_class = np.random.rand(minority_class_size, 2) + np.array([1, 1])
majority_class = np.random.rand(majority_class_size, 2)

# Calculate class proportion
class_proportion = majority_class_size // minority_class_size

# Randomly sample majority class instances
sampled_majority_class_indices = np.random.choice(majority_class_size, minority_class_size * class_proportion, replace=False)
sampled_majority_class = majority_class[sampled_majority_class_indices]

# Combine minority and sampled majority class instances
balanced_data = np.vstack((minority_class, sampled_majority_class))
labels = np.hstack((np.ones(minority_class_size), np.zeros(minority_class_size * class_proportion)))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(balanced_data, labels, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"AUC-ROC: {roc_auc:.2f}")
