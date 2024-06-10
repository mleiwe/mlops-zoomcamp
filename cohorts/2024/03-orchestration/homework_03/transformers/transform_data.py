import pandas as pd
import requests
from io import BytesIO 
from typing import List
if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform_dataframe(df, **kwargs) -> pd.DataFrame:

    #df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    #df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df