import requests
import pandas as pd

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

YEAR = "2023"
MONTH = "05"
categorical = ['PULocationID', 'DOLocationID']

url_address = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{YEAR}-{MONTH}.parquet'
print(f"Downloading dataset from {url_address}")
df = read_data(url_address)
features = df[categorical].to_dict(orient='records')
print("Now sending Request")
model_url = 'http://127.0.0.1:9696/predict' 
requests.post(model_url, json=features)