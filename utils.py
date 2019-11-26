import numpy as np
import pandas as pd
import math

from sklearn.metrics import mean_squared_error
from eli5.formatters.as_dataframe import explain_weights_df


def rmse(y_true: np.array, y_pred: np.array):
    """np.array: Root mean square error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calc_metrics(y_true, y_pred, metrics):
    """Generate metrics dataframe

    Args:

    Returns:
    """
    results = []

    for n in range(len(y_true)):
        step_results = []
        for metric in metrics:
            step_results.append(metric(y_true[n], y_pred[n]))
        results.append(step_results)

    return pd.DataFrame(results, columns=[metric.__name__ for metric in metrics])


def mean_fold_feature_weights(boost_list: list, save_path=None):
    """Generate dataframe with mearn feature weights, using eli5 library"""
    df = None

    for alg in boost_list:
        step_df = explain_weights_df(alg)
        if df is None:
            df = step_df
        else:
            df = df.append(step_df)

    df = (
        df.groupby(by="feature")["weight"]
        .mean()
        .reset_index()
        .sort_values(by="weight", ascending=False)
    )

    if save_path is not None:
        df.to_csv(save_path, index=False)
    return df


def parse_time(time_str):
    """Parse datetime from organaizers dataframe format"""
    time, halfday = time_str.split(" ")
    hours, minutes, seconds = [int(i) for i in time.split(":")]

    if halfday == "PM" and hours < 12:
        hours += 12

    return pd.Timestamp(
        year=2019, month=1, day=1, hour=hours, minute=minutes, second=seconds
    )


def q_diff(values, q1: float = 0.25, q2: float = 0.75):
    """Return diff between q2 and q1 quantiles."""
    return np.quantile(values, q2) - np.quantile(values, q1)


def mean_to_median(values):
    """Return mean / clipped median."""
    return np.mean(values) / np.clip(np.median(values), 1, None)


# copy paste from https://gist.github.com/jeromer/2005586
def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (
        math.sin(lat1) * math.cos(lat2) * math.cos(diffLong)
    )

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing
