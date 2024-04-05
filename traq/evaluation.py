import inspect
import multiprocessing as mp
from functools import partial
from math import floor, sqrt

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from traq.utils import preprocess_plate


def _build_hyperparameters(grid):
    # TODO implement - ok for now if just using default hyps
    # TODO when implementing - use OrderedD]ct
    return [{}]


def roc_auc_ci(y_true, y_score, alpha=0.05, positive=1):
    """
    Normal approximation to get a confidence interval.

    From: https://gist.github.com/doraneko94/e24643136cfb8baf03ef8a314ab9615c
    """
    z = sp.stats.norm.ppf(1 - alpha / 2)

    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2 * AUC**2 / (1 + AUC)
    SE_AUC = sqrt(
        (AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC**2) + (N2 - 1) * (Q2 - AUC**2))
        / (N1 * N2)
    )
    lower = AUC - z * SE_AUC
    upper = AUC + z * SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1

    return (lower, upper)


def precision_recall_at_k(y, yhat, k=None, k_frac=None):
    """
    Compute precision@k and recall@k.

    Params:
        y: The labels.
        yhat: The predicted probabilities.

    Returns:
        precision: The precision score.
        recall: The recall score.
    """
    yhat = np.array(yhat)

    if k is None:
        if k_frac is None:
            k = sum(y)
        else:
            k = floor(len(y) * k_frac)
    elif k >= len(y):
        k = len(y) - 1

    min_score = yhat[np.argpartition(-yhat, k)[k]]
    ypred = yhat > min_score

    return precision_score(y, ypred), recall_score(y, ypred)


def _evaluation_metrics(yhat, y, correction_factor=1):
    if len(np.unique(y)) < 2:
        return {"error": "no meaningful labels"}
    if (~np.isinf(yhat)).sum() == 0:
        return {"error": "all infinity predictions"}
    yhat[np.isinf(yhat) & (yhat > 0)] = np.max(yhat[~np.isinf(yhat)])
    yhat[np.isinf(yhat) & (yhat < 0)] = np.min(yhat[~np.isinf(yhat)])
    if np.isnan(yhat).sum() > 0:
        if np.isnan(yhat).sum() == len(yhat):
            return {"error": "missing predictions"}
        yhat[np.isnan(yhat)] = np.nanmean(yhat)

    auroc_lower, auroc_upper = roc_auc_ci(y, yhat)
    auroc_lower_corrected, auroc_upper_corrected = roc_auc_ci(
        y, yhat, alpha=0.05 / correction_factor
    )
    patk, _ = precision_recall_at_k(y, yhat)
    pat5, rat5 = precision_recall_at_k(y, yhat, 5)
    pat10, rat10 = precision_recall_at_k(y, yhat, 10)
    pat5pct, rat5pct = precision_recall_at_k(y, yhat, k_frac=0.05)
    pat10pct, rat10pct = precision_recall_at_k(y, yhat, k_frac=0.1)

    return {
        "anomaly_proportion": y.mean(),
        "auroc": roc_auc_score(y, yhat),
        "auroc.lower": auroc_lower,
        "auroc.upper": auroc_upper,
        "auroc.lower.corrected": auroc_lower_corrected,
        "auroc.upper.corrected": auroc_upper_corrected,
        "aupr": average_precision_score(y, yhat),
        "precision_at_k": patk,
        "precision_at_5": pat5,
        "recall_at_5": rat5,
        "precision_at_10": pat10,
        "recall_at_10": rat10,
        "precision_at_5pct": pat5pct,
        "recall_at_5pct": rat5pct,
        "precision_at_10pct": pat10pct,
        "recall_at_10pct": rat10pct,
    }


def evaluate_pyod_models_plate(plate, cash, trial_name, snapshot_name):
    plate_results = []
    yhats = {}

    preprocessed = preprocess_plate(plate)
    if not preprocessed:
        return None, {}
    X, y = preprocessed

    num_algorithms = 0
    for _, hyperparameter_grid in cash.items():
        num_algorithms += len(_build_hyperparameters(hyperparameter_grid))

    for algorithm, hyperparameter_grid in cash.items():
        hyperparameter_combinations = _build_hyperparameters(hyperparameter_grid)
        for hyperparameter_combination in hyperparameter_combinations:
            # Best effort at seeding given how PyOD is set up.
            np.random.seed(42)
            if "random_state" in inspect.getfullargspec(algorithm).args:
                hyperparameter_combination["random_state"] = 42

            model_instance = algorithm(**hyperparameter_combination)
            model_instance.fit(X)
            predictions = model_instance.decision_scores_
            metrics = _evaluation_metrics(
                predictions, y, correction_factor=num_algorithms
            )

            plate_results.append(
                {
                    "plate": plate.name,
                    "num_samples": X.shape[0],
                    "num_columns": X.shape[1],
                    "num_anomalies": y.sum(),
                    "anomaly_proportion": y.mean(),
                    "algorithm": algorithm.__name__,
                    "hyperparameters": hyperparameter_combination,
                    **metrics,
                }
            )
            # TODO: only applicable with no hyperparameters
            yhats[algorithm] = predictions

    # # Ensemble approach
    # scaled_predictions = []
    # for algorithm_name, predictions in yhats.items():
    #     if any(ensemble_algorithm in str(algorithm_name)
    #            for ensemble_algorithm in ("IForest", "ECOD")):
    #         scale = np.nanmax(predictions) - np.nanmin(predictions)
    #         scaled_predictions.append((predictions - np.nanmin(predictions)) / scale)
    # ensemble_predictions = np.nanmean(np.array(scaled_predictions), axis=0)
    # yhats["ensemble"] = ensemble_predictions
    # metrics = _evaluation_metrics(
    #     ensemble_predictions, y, correction_factor=num_algorithms + 1
    # )
    # plate_results.append(
    #     {
    #         "plate": plate.name,
    #         "num_samples": X.shape[0],
    #         "num_columns": X.shape[1],
    #         "num_anomalies": y.sum(),
    #         "anomaly_proportion": y.mean(),
    #         "algorithm": "ensemble",
    #         "hyperparameters": None,
    #         **metrics,
    #     }
    # )

    return plate_results, {plate.name: yhats}


def evaluate_pyod_models(dataset, cash, trial_name, num_workers=48):
    """
    Params:
        dataset: The dataset to evaluate PyOD models.
        cash: The CASH space specifying the models.
    """
    results = []
    yhats = {}
    with mp.Pool(num_workers) as pool:
        f = partial(
            evaluate_pyod_models_plate,
            cash=cash,
            trial_name=trial_name,
            snapshot_name=dataset._name,
        )
        plates = dataset.plates()
        for plate_results, yhats_plate in tqdm(
            pool.imap_unordered(f, plates), total=len(plates)
        ):
            if plate_results is not None:
                results += plate_results
                yhats = {**yhats, **yhats_plate}
    results_df = pd.DataFrame(results)

    return results_df, yhats
