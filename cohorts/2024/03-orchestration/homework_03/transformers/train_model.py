from typing import Tuple
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def linear_regression(data, *args, **kwargs) -> Tuple[BaseEstimator, BaseEstimator]:
    """
    Perform a linear regression on the data, returning the model and the dictionary vectorizer

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    X, y, dv = data
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    
    print(lin_reg.intercept_)
    
    return lin_reg, dv