#!/usr/bin/env python
# coding: utf-8
"""
Create a parquet file containing the appropriate fields locally from the yellow trip data with the specified year, and month.
"""
import sys
import pickle
import pandas as pd
import numpy as np

from datetime import datetime

def dt(hour, minute, second=0):
    """return the datetime, fixed to 1st Jan"""
    return datetime(2023, 1, 1, hour, minute, second)

def ts(hour,minute,second=0):
    """return timestamp, fixed to 1st Jan"""
    return pd.Timestamp(year=2023, month=1, day=1, hour=hour, minute=minute, second=second)

def read_data():
    """
    Reads the parquet file from the variable filename as a dataframe
    """
    df = pd.read_parquet('/Users/marcusleiwe/Documents/GitHubRepos/mlops-zoomcamp/cohorts/2024/06-best-practices/homework/tests/test.parquet')
    return df
    

def test_transform_data():
    """
    transform data prepares the data frame by
    1) Creating a duration column to record the actual ride duration in minutes
    2) Filtering out any rides shorter than 1 min, and longer than 60mins
    3) Convert categorical columns into strings
    """
    #df = pd.read_parquet('./test.parquet')

    categorical = ['PULocationID', 'DOLocationID']
    
    #Create the test data
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
        ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    actual_values = df.to_dict()
    expected_values = {
        'PULocationID': {0: '-1', 1: '1'},
        'DOLocationID': {0: '-1', 1: '1'},
        'tpep_pickup_datetime': {0: dt(1,1), 1: dt(1,2)},
        'tpep_dropoff_datetime': {0: dt(1,10), 1: dt(1,10)},
        'duration': {0: 9.0, 1: 8.0}
        }

    assert actual_values == expected_values


def test_create_parquet():
    """
    Main function, creates a parquet file with the processed data
    """
    #input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    #output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    #output_file = f'./yellow_tripdata_{year:04d}-{month:02d}.parquet'

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    df = pd.DataFrame({
        'PULocationID': {0: '-1', 1: '1'},
        'DOLocationID': {0: '-1', 1: '1'},
        'tpep_pickup_datetime': {0: dt(1,1), 1: dt(1,2)},
        'tpep_dropoff_datetime': {0: dt(1,10), 1: dt(1,10)},
        'duration': {0: 9.0, 1: 8.0}
        }
        )
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())
    year=2023
    month=1
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    actual_result = df_result.to_dict()

    expected_result = {
        'ride_id': {0: '2023/01_0', 1: '2023/01_1'},
        'predicted_duration': {0: 23.19714924577506, 1: 13.08010120625567}
        }

    ##df_result.to_parquet(output_file, engine='pyarrow', index=False)
    assert actual_result == expected_result

# if __name__=="__main__":
#     test_create_parquet()
