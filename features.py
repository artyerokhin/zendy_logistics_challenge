from utils import parse_time, calculate_initial_compass_bearing, q_diff, mean_to_median

from typing import Union, Tuple, List, Mapping, Callable
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from geopy.distance import geodesic
from h3 import h3

import numpy as np
import pandas as pd
import itertools


def fill_zeros(df: pd.DataFrame, col: str, stat: Callable):
    """Fill zero values from df col as df.col.stat values"""
    df = df.copy()

    if df[df[col] == 0].shape[0] != df.shape[0]:
        df.loc[df[col] == 0, col] = stat(df.loc[df[col] != 0, col])
        return df
    else:
        return df


def group_features(
    df: pd.DataFrame,
    group_col: Union[str, list],
    data_col: Union[str, list],
    stats: list = None,
):
    """Group features from data_col by group col with stats functions.

    Args:
        df - dataframe to group
        group_col - column(s) to group by
        data_col - column(s) that grouped
        stats - list of statistics (can be str or funcion)
    """
    if stats is None:
        stats = ["mean", "median", "std"]

    gr_df = df.groupby(by=group_col).agg({data_col: stats})

    if isinstance(group_col, list):
        gr_df.columns = [
            "{}_{}".format("_".join(group_col), "_".join(cols))
            for cols in gr_df.columns.ravel()
        ]
    else:
        gr_df.columns = [
            "{}_{}".format(group_col, "_".join(cols)) for cols in gr_df.columns.ravel()
        ]

    gr_df = gr_df.reset_index()

    return gr_df


def merge_features(df: pd.DataFrame, merge_list: List[str], fill_value: int = 0):
    """Merge dataframe with dataframes from list with value replacement
       from fill_value."""
    df = df.copy()

    for merge_df in merge_list:
        df = df.merge(merge_df, how="left", sort=False).fillna(fill_value)

    return df


def datetime_features(df: pd.DataFrame, datetime_columns: List[str], names_dict: dict):
    """Generate time feature columns (hour, week-hour, minute)"""
    df = df.copy()

    for col in datetime_columns:
        df[col + ", hour"] = df[col].dt.hour
        # weekday * 24 + hour
        df[col + ", week-hour"] = df[names_dict[col]] * 24 + df[col].dt.hour
        df[col + ", minute"] = df[col].dt.minute

    return df


def datetime_diff_features(
    df: pd.DataFrame,
    datetime_columns: List[str],
    normalization: int = 60,
    notnull: bool = True,
):
    """Generate time difference features"""
    df = df.copy()

    combinations = itertools.combinations(datetime_columns[::-1], 2)
    for first_col, second_col in combinations:
        df[f"{first_col}_{second_col}_diff"] = (
            df[first_col] - df[second_col]
        ).dt.total_seconds() / normalization

        if notnull:
            df.loc[
                df[f"{first_col}_{second_col}_diff"] < 0,
                f"{first_col}_{second_col}_diff",
            ] = 0

    return df


def geo_features(df: pd.DataFrame, point_pairs: list):
    """Generate geo features for start and end points

    Args:
        df - DataFrame
        point pairs (list) - list of point name pairs like [lat1, lon1, lat1, lon2]
    Returns:
        df - DataFrame with geo distance and bearing features
    """
    df = df.copy()

    name = "_".join(point_pairs)
    df[f"geo_distance_{name}"] = [
        geodesic((vals[0], vals[1]), (vals[2], vals[3])).meters
        for vals in df[point_pairs].values
    ]

    df[f"geo_bearing_{name}"] = [
        calculate_initial_compass_bearing((vals[0], vals[1]), (vals[2], vals[3]))
        for vals in df[point_pairs].values
    ]

    return df


def geo_hex_features(
    df: pd.DataFrame, points: List[Tuple[str, str]], res_list: List[int]
):
    """Calculate uber hex for hexagon resolutions from res_list.

    Args:
        df - dataframe
        points (list) - point column names like [Lat, Long]
        res_list (list) - list of integer resolutons
    Returns:
        df - new dataframe with hex feature columns
    """
    df = df.copy()

    for res in res_list:
        df[f"{points[0]}_{points[1]}_hex_{res}"] = [
            h3.geo_to_h3(vals[0], vals[1], res) for vals in df[points].values
        ]

    return df


def geo_hex_distance_features(
    df: pd.DataFrame, start_point_col: str, end_point_col: str
):
    """Get h3 distance between start hexagons and end hexagons

    Args:
        df - dataframe
        start_point_col (str) - start point hexagon column name
        end_point_col (str) - end point hexagon column name
    Returns:
        df - new dataframe with h3 distance column
    """
    df = df.copy()

    df[f"h3_dist_{start_point_col}_{end_point_col}"] = [
        h3.h3_distance(hexagons[0], hexagons[1])
        for hexagons in df[[f"{start_point_col}", f"{end_point_col}"]].values
    ]

    return df


def geo_cluster_features(
    df: pd.DataFrame, points: List[Tuple[str]], n_clusters: List[int]
):
    """Generate features, based on unique data points clustering

    Args:
        df - DataFrame
        points - list of point name (lat, lon) tuples
        n_clusters (list) - list of cluster sizes
    Returns:
        df - DataFrame with clustered points
    """
    df = df.copy()

    for point_pair in points:
        # fixed this for pandas
        point_pair = list(point_pair)
        for n in n_clusters:
            km = KMeans(n_clusters=n)
            sc = StandardScaler()
            name = "_".join(point_pair)

            df_dedupl = df[point_pair].drop_duplicates()
            df_dedupl[f"{name}_{n_clusters}"] = km.fit_predict(
                sc.fit_transform(df_dedupl.values)
            )
            df = df.merge(df_dedupl, how="left")

    return df


def log_features(df: pd.DataFrame, columns: List[str], clip_min: int = 1):
    """Generate logarithmed features."""
    df = df.copy()

    for col in columns:
        df[f"log_{col}"] = np.log(np.clip(df[col], clip_min, None))

    return df


def multiplication_features(df: pd.DataFrame, mult_list: List[List[str]]):
    """Multiply df features from mult_list"""
    df = df.copy()

    for mult_sublist in mult_list:
        # change tuples to lists
        mult_sublist = list(mult_sublist)
        col_name = "multiply_" + "_".join(mult_sublist)
        df.loc[:, col_name] = 1

        for n in range(len(mult_sublist)):
            df[col_name] = df[col_name] * df[mult_sublist[n]]

    return df


def division_features(
    df: pd.DataFrame, div_list: List[Tuple[str, str]], clip_min: int = 1
):
    """Generate features with division of one column value to another

    Args:
        df - DataFrame
        div_list (list) - list of (nominator, denominator) tuples
        clip_min (int) - minimum value for clipping denominator
    Returns:
        df - DataFrame with division features
    """

    for nominator, denominator in div_list:
        df[f"{nominator}_div_{denominator}"] = df[nominator] / np.clip(
            df[denominator], 1, None
        )

    return df


def process_df(df, columns, encode_cols, encoder):
    df = df[columns].copy()

    names_dict = {
        "Placement - Time": "Placement - Weekday (Mo = 1)",
        "Confirmation - Time": "Confirmation - Weekday (Mo = 1)",
        "Arrival at Pickup - Time": "Arrival at Pickup - Weekday (Mo = 1)",
        "Pickup - Time": "Pickup - Weekday (Mo = 1)",
    }

    datetime_columns = [
        "Placement - Time",
        "Confirmation - Time",
        "Arrival at Pickup - Time",
        "Pickup - Time",
    ]

    datetime_diff_feature_names = [
        f"{first_col}_{second_col}_diff"
        for first_col, second_col in itertools.combinations(datetime_columns[::-1], 2)
    ]

    for col in datetime_columns:
        df[col] = df[col].apply(parse_time)

    df = datetime_features(df, datetime_columns, names_dict)
    df = datetime_diff_features(df, datetime_columns, 60)

    df = geo_features(
        df, ["Pickup Lat", "Pickup Long", "Destination Lat", "Destination Long"]
    )

    df = division_features(
        df,
        [
            (
                "Distance (KM)",
                "geo_distance_{}".format(
                    "_".join(
                        [
                            "Pickup Lat",
                            "Pickup Long",
                            "Destination Lat",
                            "Destination Long",
                        ]
                    )
                ),
            )
        ],
    )

    for col in datetime_diff_feature_names + ["Distance (KM)"]:
        df[f"log_{col}"] = np.log(np.clip(df[col], 1, None))

    print("Dist features done")

    df = geo_hex_features(df, ["Pickup Lat", "Pickup Long"], [9, 10, 11])
    df = geo_hex_features(df, ["Destination Lat", "Destination Long"], [9, 10, 11])

    for res in [9, 10, 11]:
        start_point_name = "_".join(["Pickup Lat", "Pickup Long"]) + f"_hex_{res}"
        end_point_name = (
            "_".join(["Destination Lat", "Destination Long"]) + f"_hex_{res}"
        )
        df = geo_hex_distance_features(df, start_point_name, end_point_name)

    print("Hex features done")

    df = geo_cluster_features(
        df,
        [("Pickup Lat", "Pickup Long"), ("Destination Lat", "Destination Long")],
        [100, 200, 300, 500],
    )

    print("Cluster features done")

    for enc_col in encode_cols:
        df[enc_col] = encoder.fit_transform(df[enc_col])

    print("Encoding features done")

    # feature names are generated like in time diff features function
    division_feature_names = [
        (first_col, second_col)
        for first_col, second_col in itertools.combinations(
            datetime_diff_feature_names, 2
        )
    ]

    # itertools combination
    multiplication_feature_names = [
        f"{nominator}_div_{denominator}"
        for nominator, denominator in division_feature_names
    ]

    multiplication_feature_names_comb = list(
        itertools.product(multiplication_feature_names, multiplication_feature_names)
    )

    df = division_features(df, division_feature_names)
    df = multiplication_features(df, multiplication_feature_names_comb)

    print("All features done")
    return df


def generate_datasets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    riders: pd.DataFrame,
    names: list,
    encoding_columns: list,
    drop_columns: list,
):
    """Generate train and test datasets for competition"""
    label_encoder = LabelEncoder()

    data = train.append(test, sort=False)
    data = (
        process_df(data, names, encoding_columns, label_encoder)
        .merge(riders, how="left", on="Rider Id", sort=False)
        .fillna(0)
    )
    print("processed merged train and test")

    train["speed"] = train["Distance (KM)"] / (
        (train["Time from Pickup to Arrival"] + 1) / 3600
    )
    # fmt: off
    train["Pickup - Time, hour"] = data[:len(train)]["Pickup - Time, hour"]
    train["Personal or Business"] = data[:len(train)]["Personal or Business"]
    train["Platform Type"] = data[:len(train)]["Platform Type"]
    # fmt: on

    data = division_features(
        data,
        [
            ("No_Of_Orders", "Age"),
            ("Average_Rating", "No_of_Ratings"),
            ("No_of_Ratings", "No_Of_Orders"),
        ],
    )

    gr_list = []
    for group_col in [
        "Pickup - Day of Month",
        "Pickup - Weekday (Mo = 1)",
        "Pickup - Time, hour",
        "Rider Id",
        ["Pickup - Weekday (Mo = 1)", "Pickup - Time, hour"],
        "Personal or Business",
        "Platform Type",
    ]:

        gr_list.append(
            group_features(
                train,
                group_col=group_col,
                data_col="speed",
                stats=["mean", "median", "std", q_diff, mean_to_median],
            ).fillna(0)
        )
    print("categorical speed groupings ready")

    for n, group_col in enumerate(
        [["Pickup Lat", "Pickup Long"], ["Destination Lat", "Destination Long"]]
    ):
        place_speed = group_features(
            train[train["speed"] > 100],
            group_col,
            "speed",
            ["median", "mean", "std", "count", q_diff],
        )
        place_speed = place_speed.loc[
            place_speed["{}_{}_count".format("_".join(group_col), "speed")] > 5, :
        ]

        place_cnt = group_features(train, group_col, "Order No", ["count"])
        place_cnt = place_cnt.loc[
            place_cnt["{}_{}_count".format("_".join(group_col), "Order No")] > 5, :
        ]

        place_speed = place_speed.merge(place_cnt)

        gr_list.append(place_speed)
    print("place speed groupings ready")

    datetime_columns = [
        "Placement - Time",
        "Confirmation - Time",
        "Arrival at Pickup - Time",
        "Pickup - Time",
    ]

    datetime_diff_feature_names = [
        f"{first_col}_{second_col}_diff"
        for first_col, second_col in itertools.combinations(datetime_columns[::-1], 2)
    ]

    # fmt: off
    for data_col in datetime_diff_feature_names + [
        "geo_distance_{}".format(
            "_".join(
                [
                    "Pickup Lat",
                    "Pickup Long",
                    "Destination Lat",
                    "Destination Long",
                ])),
        "Distance (KM)"
    ]:
        gr_list.append(
            group_features(
                data[:len(train)],
                group_col="Rider Id",
                data_col=data_col,
                stats=["mean", "median", "std", q_diff],
            ).fillna(0)
        )
    # fmt: on

    # speedsters are drivers who have any over speeds
    speedsters = (
        train[train["speed"] > 100]
        .groupby(by="Rider Id")["Order No"]
        .count()
        .reset_index()
        .rename(columns={"Order No": "failed_orders_cnt"})
    )

    speedsters_stat = (
        train.groupby(by="Rider Id")["Order No"]
        .count()
        .reset_index()
        .rename(columns={"Order No": "orders_cnt"})
        .merge(speedsters)
    )

    speedsters_stat = division_features(
        speedsters_stat, [("failed_orders_cnt", "orders_cnt")]
    )

    # fmt: off
    train_data = data[:len(train)].drop(drop_columns, axis=1)
    test_data = data[len(train):].drop(drop_columns, axis=1)
    # fmt: on

    train_data = merge_features(train_data, gr_list)
    test_data = merge_features(test_data, gr_list)

    train_data = (
        train_data.merge(speedsters_stat, how="left").fillna(0).drop("Rider Id", axis=1)
    )
    test_data = (
        test_data.merge(speedsters_stat, how="left").fillna(0).drop("Rider Id", axis=1)
    )

    train_data = fill_zeros(train_data, "Rider Id_speed_median", np.median)
    test_data = fill_zeros(test_data, "Rider Id_speed_median", np.median)

    target = train["Time from Pickup to Arrival"]

    return train_data, test_data, target
