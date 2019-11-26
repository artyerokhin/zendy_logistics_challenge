import training
import utils
import features

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold

import argparse
import logging
import json


log_file = "./logfile.log"
log_level = logging.DEBUG
logging.basicConfig(
    level=log_level,
    filename=log_file,
    filemode="w+",
    format="%(asctime)-15s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("base_logger")


TRAIN_FILE = "Train.csv"
TEST_FILE = "Test.csv"
RIDERS_FILE = "Riders.csv"
SAMPLE_SUBMISSION = "SampleSubmission.csv"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_file",
    help="File to save predictions result",
    type=str,
    default="baseline.csv",
)
parser.add_argument(
    "--n_splits", default=10, help="number of cross validation splits", type=int,
)
parser.add_argument(
    "--n_params",
    default=20,
    help="number of random parameters to random search",
    type=int,
)
parser.add_argument(
    "--json_file",
    default=None,
    help="path to json file with model parameters",
    type=str,
)
parser.add_argument(
    "--explain_file",
    default=None,
    help="path to file where we save mean feature weights from all folds",
    type=str,
)
args = parser.parse_args()


if __name__ == "__main__":
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    riders = pd.read_csv(RIDERS_FILE)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION)

    names = set(train.columns) & set(test.columns)
    enc_columns = (
        ["Personal or Business"]
        + [f"Pickup Lat_Pickup Long_hex_{res}" for res in [9, 10, 11]]
        + [f"Destination Lat_Destination Long_hex_{res}" for res in [9, 10, 11]]
    )
    drop_columns = [
        "Order No",
        "User Id",
        "Placement - Time",
        "Confirmation - Time",
        "Pickup - Time",
        "Arrival at Pickup - Time",
        "Vehicle Type",
        "Confirmation - Day of Month",
        "Confirmation - Weekday (Mo = 1)",
        "Arrival at Pickup - Day of Month",
        "Arrival at Pickup - Weekday (Mo = 1)",
        "Placement - Day of Month",
        "Placement - Weekday (Mo = 1)",
    ]

    train_data, test_data, target = features.generate_datasets(
        train, test, riders, names, enc_columns, drop_columns
    )

    if args.json_file is None:
        params = training.generate_params(args.n_params)
    else:
        with open(f"{args.json_file}") as json_file:
            params = [json.load(json_file)]

    kf = KFold(n_splits=args.n_splits, shuffle=True)
    best_score, best_params, best_alg_list = training.find_best_algo(
        train_data, target, lgb.LGBMRegressor, kf, params, "rmse", 42, 20
    )

    print("best score:", best_score)
    print("best params:", best_params)

    logger.debug(f"best score: {best_score}")
    logger.debug(f"best paraams: {best_params}")

    if args.explain_file is not None:
        explain_df = utils.mean_fold_feature_weights(best_alg_list, args.explain_file)

    sample_submission["Time from Pickup to Arrival"] = np.mean(
        [alg_elem.predict(test_data) for alg_elem in best_alg_list], axis=0
    )
    sample_submission.to_csv(f"{args.save_file}", index=False)
