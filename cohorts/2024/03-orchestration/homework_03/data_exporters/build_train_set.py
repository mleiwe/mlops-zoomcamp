from typing import Tuple
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_exporter
def dict_vectoriser(df, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
    """
    This code outputs the dataframe transformed into a dictionary vectoriser as well as the target variable. 

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    
    categorical = ['PULocationID', 'DOLocationID']
    target = kwargs.get('target')

    df_slim = df[categorical]
    X_dict = df_slim.to_dict(orient='records')
    dv = DictVectorizer()
    X = dv.fit_transform(X_dict)

    y = df[target]

    return X, y, dv


@test
def test_output_columns(X, y, dv) -> None:
    assert (
        y.dtype == 'float64'
    ), f'y dtype is {y.dtype} not float64'
    assert (
        X.shape[0] == 3316216
    ), f'There are {X.shape[0]} records not 3316216'
    assert (
        X.shape[1] == 518
    ), f'There are {X.shape[1]} columns not 518'
    assert (
        len(y) == X.shape[0]
    ), f'The length of X and y do not match'
    