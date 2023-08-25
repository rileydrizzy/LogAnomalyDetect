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


"""doc
"""
from functools import partial

import tensorflow as tf


def build_model(vocab_size, embed_dim, Sequnce_length):
    """1DCNN doc

    Parameters
    ----------
    file_path : str

    Returns
    -------
    model : object
        model
    """

    input_ = tf.keras.layers.Input(shape=(Sequnce_length,))
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, 
                                                output_dim=embed_dim, mask_zero= True)
    conv1D = tf.keras.layers.Conv1D(filters=10, kernel_size= 2)
    pool = tf.keras.layers.MaxPool1D()
    flatten = tf.keras.layers.GlobalAveragePooling1D()
    drop1 = tf.keras.layers.Dropout(0.5)
    dense_layer = tf.keras.layers.Dense(units =100, activation='relu')
    drop2 = tf.keras.layers.Dropout(0.5)
    output_layer = tf.keras.layers.Dense(1,activation='sigmoid')

    model = tf.keras.Sequential([input,embedding_layer,conv1D,pool,flatten,drop1,dense_layer,drop2,output_layer])
    return model



# !Alerts


"""doc
"""

import mlflow
import tensorflow as tf

from utils.utils import get_dataset, get_tokenizer, set_seed, tracking

# TODO SET SEED
set_seed()
# TODO LOSS, OPTIM
LOSS = 'temp'
OPTIM = 'temp'
tensorb = tf.keras.callbacks.TensorBoard()
callback_list = ''
experiment_id = tracking('TEST RUN')

# ! START HERE
train_data = get_dataset(file_path='data',shuffle= True)
valid_data = get_dataset(file_path='data')
tokenizer, vocab_size = get_tokenizer(dataset=train_data)

# TODO Model is been imported
model_name = '1DCNN'
main_model = tf.keras.Sequential([tf.keras.layers.Dense(10),
                                  tf.keras.layers.Dense(1)])

main_model.compile(loss=LOSS, optimizer=OPTIM, metrics=["f1_score"])

mlflow.tensorflow.autolog(log_datasets=False,)
with mlflow.start_run(run_name=model_name, experiment_id=experiment_id,):
    # TODO Setup distributed training
    # # TODO Setup train_time
