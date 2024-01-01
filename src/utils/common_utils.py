"""Utility Functions Module

Functions:
- set_seed(seed=42): Set random seed for reproducibility.
- set_mlflow_tracking(model_name): Set up MLflow experiment tracking.
- tensorboard_dir(model_name): Generate directory for TensorBoard logs.
- get_device_strategy(): Get TensorFlow distribution strategy based on available devices.
- plot_precision_recall_curve(model, eval_datatset, save_path=None, save_to_mlflow=False):\
    Generate and plot the precision-recall curve for a TensorFlow model.
- plot_confusion_matrix(model, eval_datatset, threshold=0.5, save_path=None, save_to_mlflow=False):\
    Generate and plot the confusion matrix for a TensorFlow model.

"""

import os
from pathlib import Path
from time import strftime

import mlflow
import seaborn as sns
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import confusion_matrix


def set_seed(seed=42):
    """Set random seed for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Random seed value, by default 42.
    """
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_mlflow_tracking(model_name):
    """Set up MLflow experiment tracking.

    Parameters
    ----------
    model_name : str
        Name of the model.

    Returns
    -------
    int
        Experiment ID for MLflow tracking.

    Notes
    -----
    The function assumes a DagsHub-specific MLflow tracking URI.
    """
    mlflow.set_tracking_uri("https://dagshub.com/rileydrizzy/log_anomaly.mlflow")
    experiment = mlflow.get_experiment_by_name(model_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(model_name)
        return experiment_id
    return experiment.experiment_id


def tensorboard_dir(model_name):
    """Generate directory for TensorBoard logs.

    Parameters
    ----------
    model_name : str
        Name of the model.

    Returns
    -------
    pathlib.Path
        Path to the directory for TensorBoard logs.
    """
    model_directory = "Tensorboard_logs/" + model_name
    runs = strftime("run_%Y_%m_%d_%H_%M_%S")
    log_dir = Path(model_directory, runs)
    return log_dir


def get_device_strategy():
    """Get TensorFlow distribution strategy based on available devices.

    Returns
    -------
    tuple
        - tf.distribute.Strategy: TensorFlow distribution strategy.
        - str: Description of the available device ('GPU' or 'CPU').
    """
    if tf.test.gpu_device_name():
        strategy = tf.distribute.MirroredStrategy()
        return (strategy, "GPU")
    strategy = tf.distribute.OneDeviceStrategy(device="/device:CPU:0")
    return (strategy, "CPU")


def plot_precision_recall_curve(
    model, eval_dataset, save_path=None, save_to_mlflow=False
):
    """Generate and plot the precision-recall curve for a TensorFlow model.

    Parameters
    ----------
    model : tf.keras.Model
        The trained TensorFlow model.
    eval_dataset_ : tf.data.Dataset
        The dataset to perform the model evalution on
    save_path : str, Path, optional
        Optional path to save the plot as an image, by default None, by default None
    save_to_mlflow: bool, optional
        To save the generated figure to mlflow, by default False.

    Raises
    ------
    ValueError
        Raised when input data is empty
    """

    # Convert TensorFlow Dataset to NumPy arrays
    x_test, y_test = zip(*eval_dataset)

    # Stack the batches of features and label vertically
    x_test = np.vstack(list(x_test))
    y_test = np.vstack(list(y_test))

    # Error handling for input shapes, if needed
    if len(x_test) == 0 or len(y_test) == 0:
        raise ValueError("Input data is empty.")

    y_pred = model.predict(x_test)

    # flatten the true labels
    y_test = np.reshape(y_test, (-1,))
    y_pred = np.reshape(y_pred, (-1,))

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

    auc_score = auc(recall, precision)

    # Plot the precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        label=f"Precision-Recall Curve (AUC = {auc_score:.2f})",
        color="b",
    )

    # Plot points on the curve for different thresholds
    for i, threshold in enumerate(np.linspace(0, 1, 10)):
        index = np.argmax(thresholds >= threshold)
        plt.scatter(
            recall[index],
            precision[index],
            marker="o",
            color="red",
            label=f"Threshold = {threshold:.2f}" if i == 0 else "",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve with Threshold Points")
    plt.legend(loc="upper right")
    plt.grid(True)

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Precision-Recall Curve saved at: {save_path}")
        if save_to_mlflow:
            mlflow.log_artifact(save_path)
    else:
        plt.show()


def plot_confusion_matrix(
    model, eval_dataset, threshold=0.5, save_path=None, save_to_mlflow=False
):
    """
    Generate and plot the confusion matrix for a TensorFlow model.

    Parameters
    ----------
    model : tf.keras.Model
        The trained TensorFlow model.
    eval_dataset_ : tf.data.Dataset
        The dataset to perform the model evalution on
    threshold : float, optional
        Decision threshold for binary classification, by default 0.5.
    save_path : str, optional
        Optional path to save the plot as an image.

    Raises
    ------
    ValueError
        Raised when input data is empty
    """

    # Convert TensorFlow Dataset to NumPy arrays
    x_test, y_test = zip(*eval_dataset)

    # Stack the batches of features and label vertically
    x_test = np.vstack(list(x_test))
    y_test = np.vstack(list(y_test))

    # Error handling for input shapes
    if len(x_test) == 0 or len(y_test) == 0:
        raise ValueError("Input data is empty.")

    y_pred = model.predict(x_test)

    # Apply threshold for binary classification
    y_pred_binary = (y_pred > threshold).astype(int)

    # flatten the true labels
    y_test = np.reshape(y_test, (-1,))
    cm = confusion_matrix(y_test, y_pred_binary)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 14}
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion Matrix saved at: {save_path}")
        if save_to_mlflow:
            mlflow.log_artifact(save_path)
    else:
        plt.show()
