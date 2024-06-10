if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

import pandas as pd
import requests
from io import BytesIO 
from typing import List

@data_loader
def load_data(**kwargs) -> pd.DataFrame:
    """
    Template code for loading data from any source.

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    dfs : List[pd.DataFrame] = []
    url_addresses : List = []
    for year, months in [(2023, (3,4))]: #limited to March 2023
        for i in range(*months):
            url_address = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_'f'{year}-{i:02d}.parquet'
            response = requests.get(url_address)
            if response.status_code != 200:
                raise Exception(response.text) #Display the request's error code if it doesn't work
            df = pd.read_parquet(BytesIO(response.content))
            dfs.append(df)
            
    return dfs