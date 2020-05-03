from typing import Union, Tuple, List, Mapping
from utils import calc_metrics, rmse
from sklearn.metrics import mean_absolute_error, mean_squared_error

import pandas as pd
import numpy as np
from tqdm import tqdm


def generate_params(n_generations: int = 20):
    """Generate random parameters for algorithm"""
    params = []

    for _ in range(n_generations):
        params.append(
            {
                "learning_rate": np.random.randint(1, 30) / 100,
                "max_depth": np.random.randint(-1, 25),
                "n_estimators": np.random.randint(100, 500),
                "num_leaves": np.random.choice(
                    [7, 15, 21, 27, 31, 61, 81, 127, 197, 231, 275, 302]
                ),
                "max_bin": np.random.choice([3, 5, 10, 12, 18, 20, 22]),
                "boosting_type": np.random.choice(["gbdt"]),
                "subsample": np.random.choice([0.5, 0.7, 0.8, 0.9, 1]),
                "colsample_bytree": np.random.choice([0.5, 0.7, 0.8, 0.9, 1]),
                "reg_alpha": np.random.randint(0, 100) / 10,
                "reg_lambda": np.random.randint(0, 100) / 10,
            }
        )

    return params


def find_best_algo(
    train_data: pd.DataFrame,
    target: Union[np.array, pd.Series],
    alg_class,
    cross_val,
    params: List[Mapping],
    metric: str,
    random_state: int,
    early_stopping: int,
) -> Tuple[float, dict, list]:
    """"Find best algorithm

    Args:
        train_data: training data
        target: target values
        alg_class: algorith class with fit and early_stopping
        cross_val: sklearn cross validation class object
        params: list of parameters dicts
        metric: optimization metric name
        random_state: random seed for algorithm
        early_stopping: number of early stopping rounds

    Return:
        best_score (float): best metric score
        best_params (dict): best algorithm parameters
        best_alg_list (list): list of trained on validation splits algorithms
    """
    best_params = None
    best_score = None
    best_alg_list = None

    for param in tqdm(params):
        real_values = []
        predictions = []
        alg_list = []
        param["random_state"] = random_state

        for train_index, test_index in cross_val.split(train_data):
            alg = alg_class(**param)

            X_train, X_test = train_data.iloc[train_index], train_data.iloc[test_index]
            y_train, y_test = (
                target.iloc[train_index].values,
                target.iloc[test_index].values,
            )

            if param.get("boosting_type", None) != "dart" and param.get(
                "boosting_type", None
            ):
                alg.fit(
                    X_train,
                    y_train,
                    early_stopping_rounds=early_stopping,
                    eval_set=[(X_test, y_test)],
                    verbose=False,
                )
            else:
                alg.fit(X_train, y_train)

            alg_list.append(alg)
            predictions.append(alg.predict(X_test))
            real_values.append(y_test)

        metrics_df = calc_metrics(
            real_values, predictions, [mean_absolute_error, mean_squared_error, rmse]
        )

        if best_score is None or metrics_df[metric].mean() < best_score:
            print(
                "new best score {}+-{:.2f}".format(
                    metrics_df[metric].mean(), metrics_df[metric].std()
                )
            )
            best_score = metrics_df[metric].mean()
            best_params = param
            best_alg_list = alg_list

    return best_score, best_params, best_alg_list
