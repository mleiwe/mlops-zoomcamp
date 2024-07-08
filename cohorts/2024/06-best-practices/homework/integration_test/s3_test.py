#!/usr/bin/env python
# coding: utf-8
import pandas as pd



options = {
    'client_kwargs': {
        'endpoint_url': S3_ENDPOINT_URL
    }
}
def read_data(filename):
    """
    Reads the parquet file from the variable filename as a dataframe
    """
    df = pd.read_parquet(filename)
    return df

def test_transform_data(df):
    """
    transform data prepares the data frame by
    1) Creating a duration column to record the actual ride duration in minutes
    2) Filtering out any rides shorter than 1 min, and longer than 60mins
    3) Convert categorical columns into strings
    """
    categorical = ['PULocationID', 'DOLocationID']
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def create_parquet(df,output_file):
    """
    Main function, creates a parquet file with the processed data
    """
    #input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    #output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    #output_file = f'./yellow_tripdata_{year:04d}-{month:02d}.parquet'

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

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
    
    df_result.to_parquet(output_file, engine='pyarrow', index=False)

#df = pd.read_parquet('s3://bucket/file.parquet', storage_options=options)

def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    # rest of the main function ... 
    df = read_data(input_file)
    df = test_transform_data(df)
    create_parquet(df,output_file)

