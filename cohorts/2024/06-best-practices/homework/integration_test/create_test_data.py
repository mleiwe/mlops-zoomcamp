import pandas as pd
import boto3
import s3fs
from datetime import datetime

def dt(hour, minute, second=0):
    """return the datetime, fixed to 1st Jan"""
    return datetime(2023, 1, 1, hour, minute, second)

def create_data_frame():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    return df

def save_dataframe_in_S3(df_input,input_file, options):
    df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

def save_dataframe(df_input,input_file):
    df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    #storage_options=options
)

def test_create_parquet():
    df = create_data_frame()

    # Configure the S3 client to use LocalStack
    s3_endpoint_url = 'http://localhost:4566'
    s3_client = boto3.client('s3', endpoint_url=s3_endpoint_url)

    # Create an S3 filesystem object
    #s3 = s3fs.S3FileSystem(client_kwargs={'endpoint_url': s3_endpoint_url})

    # Specify the S3 bucket and file path
    bucket_name = 'nyc-duration'
    file_path = f's3://{bucket_name}/test_input_data.parquet'
    options = {'s3':s3_client}
    
    #save_dataframe_in_S3(df,file_path,options)
    filename = "test_input_data.parquet"
    save_dataframe(df,filename)
    print(f"DataFrame written to {file_path}")

if __name__=="__main__":
    test_create_parquet()