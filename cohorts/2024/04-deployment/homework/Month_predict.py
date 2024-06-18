import pickle
from flask import Flask, request, jsonify
import pandas as pd

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

YEAR = "2023"
MONTH = "05"

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def predict(YEAR,MONTH):
    print("Running predict function")
    url_address = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{YEAR}-{MONTH}.parquet'
    print(f"Downloading dataset from {url_address}")
    df = read_data(url_address)
    print("Transforming data")
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    print("Making Predictions")
    y_pred = model.predict(X_val)

    yr = int(YEAR)
    mnth = int(MONTH)
    df['ride_id'] = f'{yr:04d}/{mnth:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })
    # df_result.to_parquet(
    #     'df_result.parquet',
    #     engine='pyarrow',
    #     compression=None,
    #     index=False
    # )

    ##print the mean predicted duration
    print(f"The mean predicted duration is {df_result['predicted_duration'].mean():.2f}")
    return df_result['predicted_duration'].mean()

predict(YEAR,MONTH)