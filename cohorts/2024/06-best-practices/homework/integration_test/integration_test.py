#!/usr/bin/env python
# coding: utf-8
"""
Create a parquet file containing the appropriate fields locally from the yellow trip data with the specified year, and month.
"""
import sys
import pickle
import pandas as pd
import os
from datetime import datetime

def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


# def read_data(filename:str, categorical:list) -> pd.DataFrame:
#     """
#     Reads the parquet file from the variable filename as a dataframe, then converts the categorical columns into a string
#     """
#     options = {
#     'client_kwargs': {
#         'endpoint_url': os.getenv('S3_ENDPOINT_URL',https://localhost:4566),
#         },
#     }

def dt(hour, minute, second=0):
    """return the datetime, fixed to 1st Jan"""
    return datetime(2023, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
]

input_file = get_input_path(2023, 1)
columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

options = {
    'client_kwargs': {
        'endpoint_url': os.getenv('S3_ENDPOINT_URL','https://localhost:4566')
        }
    }
df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)
