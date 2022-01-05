import numpy as np
import tensorflow as tf
from sklearn.metrics import top_k_accuracy_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ############################### tf based metrics #############################

def CategoricalAccuracy(**kwargs):
    return tf.keras.metrics.CategoricalAccuracy(**kwargs)


def TopKCategoricalAccuracy(**kwargs):
    return tf.keras.metrics.TopKCategoricalAccuracy(**kwargs)


# ################### test dataset only uses numpy metrics #####################
# ############################## numpy based metrics ###########################


def accuracy(target: np.array, output: np.array):
    """
    Accuracy: TP + TN / (TP + TN + FP + FN)
    equivalent to:
        pred = np.argmax(output, axis=1)
        correct = np.sum(pred == target)
        return correct / len(target)
    """
    pred = np.argmax(output, axis=1)
    assert pred.shape[0] == len(target)
    return accuracy_score(target, pred)


def confusion_matrix(target: np.array, output: np.array, num_classes: int):
    """
    Accumulates the confusion matrix
    """
    conf_matrix = np.zeros([num_classes, num_classes])
    pred = np.argmax(output, 1)
    for t, p in zip(target.reshape(-1), pred.reshape(-1)):
        conf_matrix[t, p] += 1
    return conf_matrix


def precision(target: np.array, output: np.array, average: str = "weighted"):
    """
    Precision: TP / (TP + FP)
    """
    valid_avgs = {"micro", "macro", "weighted"}
    if average not in valid_avgs:
        raise ValueError(f"{average} mode is not supported. Use from {valid_avgs}")
    pred = np.argmax(output, axis=1)
    assert pred.shape[0] == len(target)
    return precision_score(target, pred, average=average)


def recall(target: np.array, output: np.array, average: str = "weighted"):
    """
    Recall: TP / (TP + FN)
    """
    valid_avgs = {"micro", "macro", "weighted"}
    if average not in valid_avgs:
        raise ValueError(f"{average} mode is not supported. Use from {valid_avgs}")
    pred = np.argmax(output, axis=1)
    assert pred.shape[0] == len(target)
    return recall_score(target, pred, average=average)


def f1score(target: np.array, output: np.array, average: str = "weighted"):
    """
    F1 score: (2 * p * r) / (p + r)
    """
    valid_avgs = {"micro", "macro", "weighted"}
    if average not in valid_avgs:
        raise ValueError(f"{average} mode is not supported. Use from {valid_avgs}")
    pred = np.argmax(output, axis=1)
    assert pred.shape[0] == len(target)
    return f1_score(target, pred, average=average)


def accuracy_mse(target: np.array, output: np.array):
    """
    Accuracy when using regression rather than classification.
    """
    assert len(output) == len(target)
    correct = np.sum(((output - target).abs() < 1))
    return correct / len(target)


def acc_per_class(target: np.array, output: np.array, num_classes: int):
    """
    Calculates acc per class
    """
    conf_mat = confusion_matrix(target, output, num_classes)
    return conf_mat.diagonal() / conf_mat.sum(1)


def top_k_acc(target: np.array, output: np.array, k: int):
    return top_k_accuracy_score(y_true=target, y_score=output, k=k)


def classification_report_sklearn(target: np.array, output: np.array, target_names: list = None):
    pred = np.argmax(output, axis=1)
    assert pred.shape[0] == len(target)
    return classification_report(target, pred, target_names=target_names)
