from typing import Dict, List, NamedTuple

import numpy as np


class EvaluationMetrics(NamedTuple):
    matrix: np.ndarray
    accuracy: float
    recall: float
    precision: float
    f1: float


class Results(NamedTuple):
    macro: EvaluationMetrics
    label: Dict[str, EvaluationMetrics]


def evaluate(predictions: np.ndarray, y: np.ndarray) -> float:
    return np.sum(predictions == y)


def confusion_matrix(predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
    labels = np.unique(y)
    matrix = np.zeros((len(labels), len(labels)))

    for i, prediction in enumerate(predictions):
        col = np.argwhere(labels == prediction)
        row = np.argwhere(labels == y[i])
        matrix[row, col] += 1

    return matrix


def confusion_matrix_by_label(
    predictions: np.ndarray, y: np.ndarray
) -> Dict[str, np.ndarray]:
    matrices = {}
    labels = np.unique(y)

    for label in labels:
        label_y = (y != label).astype(int)
        label_predictions = (predictions != label).astype(int)
        matrix = confusion_matrix(label_predictions, label_y)

        matrices[str(label)] = matrix

    return matrices


def evaluation_metrics(matrix: np.ndarray) -> EvaluationMetrics:
    accuracy = (matrix[0, 0] + matrix[1, 1]) / np.sum(matrix)

    precision = (
        matrix[0, 0] / np.sum(matrix[:, 0]) if np.sum(matrix[:, 0]) > 0 else 0
    )
    recall = (
        matrix[0, 0] / np.sum(matrix[0, :]) if np.sum(matrix[0, :]) != 0 else 0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) != 0
        else 0
    )

    return EvaluationMetrics(
        matrix=matrix,
        accuracy=accuracy,
        recall=recall,
        precision=precision,
        f1=f1,
    )


def evaluate_by_label(
    predictions: np.ndarray, y: np.ndarray
) -> Dict[str, EvaluationMetrics]:
    label_results = {}
    matrices = confusion_matrix_by_label(predictions, y)

    for label, matrix in matrices.items():
        label_results[label] = evaluation_metrics(matrix)

    return label_results


def get_mean_key_value(key, arr):
    return np.mean([i.get(key) for i in arr])


def macro_averaged_results(
    matrix: np.ndarray, label_results: Dict[str, EvaluationMetrics]
) -> EvaluationMetrics:
    results = label_results.values()

    return EvaluationMetrics(
        matrix=matrix,
        accuracy=float(np.mean([r.accuracy for r in results])),
        precision=float(np.mean([r.precision for r in results])),
        recall=float(np.mean([r.recall for r in results])),
        f1=float(np.mean([r.f1 for r in results])),
    )


def test_results(predictions: np.ndarray, y: np.ndarray) -> Results:
    matrix = confusion_matrix(predictions, y)
    label = evaluate_by_label(predictions, y)
    macro = macro_averaged_results(matrix, label)

    return Results(macro=macro, label=label)


def get_mean_evaluation_metrics(folds: List[Results]) -> EvaluationMetrics:
    results = [f.macro for f in folds]

    return EvaluationMetrics(
        matrix=np.sum([f.macro.matrix for f in folds], axis=0),
        accuracy=float(np.mean([r.accuracy for r in results])),
        precision=float(np.mean([r.precision for r in results])),
        recall=float(np.mean([r.recall for r in results])),
        f1=float(np.mean([r.f1 for r in results])),
    )
