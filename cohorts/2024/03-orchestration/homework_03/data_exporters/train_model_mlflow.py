from typing import Tuple
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from mlops.utils.logging import setup_experiment, track_experiment
import mlflow
#from mlflow import MLflow Client
import pickle

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


# HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
# RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']
# top_n = 5
# client = MlflowClient()

# # Retrieve the top_n model runs and log the models
# experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
# runs = client.search_runs(
#     experiment_ids=experiment.experiment_id,
#     run_view_type=ViewType.ACTIVE_ONLY,
#     order_by=["metrics.RMSE ASC"],
#     max_results=top_n


@data_exporter
def linear_regression(data, *args, **kwargs) -> Tuple[BaseEstimator, BaseEstimator]:
    """
    Perform a linear regression on the data, returning the model and the dictionary vectorizer

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    #MLflow params
    print("Setting up mlflow")
    EXPERIMENT_NAME = 'nyc-taxi-experiment-homework'
    TRACKING_URI = 'sqlite:///mlflow.db'
    
    #client = MLflowClient()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():      
        #Train model
        print("Training model")
        X, y, dv = data
        lin_reg = LinearRegression()
        lin_reg.fit(X, y)

        y_pred = lin_reg.predict(X)
        rmse = mean_squared_error(y, y_pred, squared=False)

        mlflow.set_tag("developer","Marcus")
        mlflow.log_metric("rmse_lin_reg",rmse)

        #Save model
        print("Saving Model")
        with open("model.pkl", "wb") as f:
            pickle.dump(lin_reg, f)
        mlflow.log_artifact("model.pkl")
        #track_experiment(experiment_name=EXPERIMENT_NAME, developer='Marcus', model=lin_reg)
    
    print(lin_reg.intercept_)
    
    return lin_reg, dv

