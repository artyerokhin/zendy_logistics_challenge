# Zindi sendy logistics [challenge](https://zindi.africa/competitions/sendy-logistics-challenge)
**Task:** predict ETA (estimated time of arrival) for bike taxi service
**Metric:** RMSE

### Final result:
* RMSE: ~708
* Place: 27/433

### Code from repository gives ~712 at private leaderbord (couldn't transfer all features from notebooks)

### Insights:
1. There was outliers in data (speed > 120 km/h), because of forgetfull drivers (3-10 seconds sessions with several km distance);
2. Additional data doesn't add much to score;
3. Target encodings showed themselfs pretty good;
4. Drivers charasteristics (median speed, ratings, etc.) was quite good features;
5. Stock lightgbm parameters really good;
6. Networkx is slow (just forget it);
7. Code refactoring from jupyter notebook was really slow this time.

# Top 10 features by weight:
|feature|description|weight|
| ---- | ---- | ---- |
|Distance_(KM)| Distance| 0.3384|
|log_Distance_(KM)| Distance logarithm| 0.1428|
|Rider_Id_speed_median| Rider median speed| 0.1103|
|geo_distance_Pickup_Lat_Pickup_Long_Destination_Lat_Destination_Long| grouped by start and end point map distance| 0.0965|
|h3_dist_Pickup_Lat_Pickup_Long_hex_11_Destination_Lat_Destination_Long_hex_11|disntace between h3 uber hexagons (resolution=11)|0.0699|
|h3_dist_Pickup_Lat_Pickup_Long_hex_10_Destination_Lat_Destination_Long_hex_10|disntace between h3 uber hexagons (resolution=10)|0.0231
|failed_orders_cnt_div_orders_cnt|share of failed drivers orders (speed > 120 km/h)|0.0161
Pickup_-_Time_Arrival_at_Pickup_-_Time_diff|time difference between pick up and arrival|0.0132
h3_dist_Pickup_Lat_Pickup_Long_hex_9_Destination_Lat_Destination_Long_hex_9|disntace between h3 uber hexagons (resolution=9)|0.0109
Destination_Lat|destination lattitude|0.0089|

How to start code:
`python -m train.py`

optional arguments:
  `-h, --help `           show this help message and exit
  `--save_file SAVE_FILE` file to save predictions result
  `--n_splits N_SPLITS`   number of cross validation splits
  `--n_params N_PARAMS`   number of random parameters to random search
  `--json_file JSON_FILE` path to json file with model parameters
  `--explain_file EXPLAIN_FILE` path to file where we save mean feature weights from all folds
