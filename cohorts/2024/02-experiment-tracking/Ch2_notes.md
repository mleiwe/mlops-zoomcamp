# MLOps Zoomcamp 2024 - Marcus' Chapter 2 notes

## 2.1 Experiment Tracking
Hosted by Christian Martinez. Basically this seems to be a talk on how to set up [MLflow](https://mlflow.org/).

### Important concepts/definitions
* `ML Experiements`: This is the process of building an ML model
* `Experiment run`: Each trial is an ML experiment
* `Run artifact`: Any file assocated with a specific ML run
* `Experiment metadata`: All the information related to the overall experiment.

### What is experiment tracking?
*Experiment tracking is the process of keeping track of all the **relevant information** from an **ML experiment*** 

This typically includes...
* Source code
* Environment
* Data
* Model
* Hyperparameters
* Metrics
* And many more

But this can vary depending on the experiment.

### Why is experiment tacking important?
* Reproducibility
* Organisation: Multiple people need to use the code or work on it so 
* Optimisation

### Don'ts
* Do not rely on GoogleSheets or Excel
    * Error Prone
    * No typical standard format; e.g. in csv it is hard to save the arrays without converting it to a string.
    * Visibility and collaboration is hard.

### What is MLflow?
MLflow is *"An open source platform for the machine learning lifecycle"*

In reality it's just a pip-installable Python package that contains four modules:
* `Tracking`: Focused on experiment tracking. 

    *"The MLflow Tracking is an API and UI for logging parameters, code versions, metrics, and output files when running your machine learning code and for later visualizing the results."* This can work beyond python, it works with REST, R, and Java APIs.

* `Models`: Types of models. 
    
    *"An MLflow Model is a standard format for packaging machine learning models that can be used in a variety of downstream tools".*
* `Model Registry`: Used to manage models. 
    
    *"The MLflow Model Registry component is a centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of an MLflow Model."*
* `Projects`: 
    *"An MLflow Project is a format for packaging data science code in a reusable and reproducible way, based primarily on conventions".* 
    
    NB This is out of scope for the course. 

More information is available within the [documentation](https://mlflow.org/docs/latest/introduction/index.html).

### MLflow tracking
MLflow tracking organises your experiment into *runs*. These runs keep track of: 
* `Parameters`: Alongside your typical input arguments etc. This can even include the path to the data you used to train/test the model, allowing you to even keep track of different preprocessing that you performed on the data
* `Scoring metrics`: Accuracy, F1 score, etc. metrics from train, test, and validation set
* `Metadata`: You can add tags to help you filter.
* `Artifacts`: Whatever outputs you deem necessary. Could even include figures, but this does come with a memory cost.
* `Models`: Sometimes it might even make sense to save the model. Especially if you are doing more than simple hyperparameter tuning.

Furthermore is also automatically logs metadata about the run including
* `Source code`
* `Version` (git commit),
* `Start` and `End` time
* `Author`

**Essentially this is information about a group of runs**

This can all be run through a simple line

    $mlflow ui

NB There are some extra things you might need in the backend e.g. PostgresSQL.

## 2.2 Getting started with MLflow
This will be a brief description on how to use MLflow for an example problem

### 2.2.1. Create a requirements.txt 
In this case we will need
* python==3.9
* mlflow
* jupyter
* sckit-learn
* panadas
* seaborn
* hyperopt
* xgboost

In this case I recommend using VSCode to create your `requirements.txt` file. 

*NB You can also have this linked to your virtual machine if you are not running locally. For more information on how to set up a GCP virtual machine see my step by step description [here](https://github.com/mleiwe/mlops-zoomcamp/blob/Ch1_Marcus/cohorts/2024/01-intro/GoogleCloudSetUpNotes.md)*

#####To Do##########

Add screenshot/video of how to create the requirements.txt file

####################

### 2.2.2 Create your virtual environment
From the `requirements.txt` file you can now create your virtual environment (venv). There are several ways in which you can do this and several articles e.g.Sam LaFell's [medium blog post](https://medium.com/@SamLaFell/why-you-need-to-ditch-pip-and-conda-61edff26f8bd) that say you should use one way or another. In my opinion the best one is the one that works best for your project and one that you are either familiar with, or you have time to learn.

Here's a vaguely helpful table (any comments and/or suggestions welcome).

|                   | venv    | conda     | miniconda     |pipenv  | poetry    |
|-------------------|---------|-----------|---------------|--------|-----------|
| **Good for...**   | **A simple project with minimal dependencies**. It's lightweight and built-in to python | **Beginners**. Conda is very user friendly, has a GUI and CLI, supports non-python packages and is consistent across platforms | When you need conda but **lightweight** | When you are **deploying to the web**. It is also reasonably user-friendly | A **python project with a range of dependencies**. It's quite modern, and user-friendly |
| **Bad for...**    | **Non-python dependencies**. I believe it struggles if needed to be used [across multiple platforms](https://stackoverflow.com/questions/12033861/cross-platform-interface-for-virtualenv) too | **efficiency** Conda is large (~2GB memory required) and can be comparatively slow                                     | **Large projects** Miniconda doesn't have the full suite of packages of conda     | **Non-python dependencies**. Has been described as a bit of a [bodge job](https://www.reddit.com/r/learnpython/comments/or1qwh/virtualenv_vs_pipenv_vs_conda_is_one_superior_to/) | **Non-python dependencies** It is also heavy compared to venv |

#### With Conda
Assuming you have conda already installed
1. Create the venv
    ```
    $conda create -n environment_name
    ```

2. Activate the venv
    ```
    $conda activate environment_name
    ```
3. Install packages from the `requirements.txt`
    ```
    $conda install --file requirements.txt
    ```
#### With Pipenv
If pipenv is already installed
1. Install and create a new environment
    ```
    $pipenv install -r path/to/requirements.txt
    ```
    NB you may also need to ceed control of versioning to the `pipfile` if you have versioning. You can either do that by altering the `requirements.txt` file or if you want to keep the versions run
    ```
    $pipenv lock --keep-outdated
    ```
2. Activate the environment
    ```
    $pipenv shell
    ```
#### With Poetry
1. Create a new poetry project
    ```
    $poetry new environment_name
    ```
2. Navigate to the environment(project) directory
    ```
    $cd environment_name
    ```
3. Install dependencies from the requirements.tt
    ```
    $poetry install --no-root -r path/to/requirements.txt
    ```
    NB the `--no-root` flag is there to ensure the dependencies are installed in the venv and not system-wide 

#### venv (for true OG style)
For true robustness and safety, or jut to show off
1. Create the virtual environment
    ```
    $python -m venv /path/to/new/virtual/environment
    ```
2. Activate the new venv
    
    For UNIX or MacOS
    ```
    $source environment_name/bin/activate
    ```

    For Windows
    ```
    $myenv\Scripts\activate
    ```
3. Install dependencies from the `requirements.txt`
    ```
    $pip install -r requirements.txt