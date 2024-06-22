# Module 4: Deployment
* Design Phase
* Train (e.g. exp tracking etc. and training pipeline)
* Now it needs to go into operation/deployment

## 4.1 Three ways of deploying a model

* **Batch Deployment**: Batch deployment is for when things do not need to be modelled immediately. Here the new data is applied to the data "regularly". Also known as *Offline Deployment*.
* **Online Deployment**: Here the model is always available. There are two sub-categories here.
    
    * **Web Services**: The model is available at a web address, you have to request it to provide an answer.
    * **Streaming**: The model is constantly listening for events on the stream and react to this.

*NB To be honest, it appears that there are no clear definitions deployment types. These notes will discuss three but some discuss [4](https://medium.com/@FrankAdams7/the-four-machine-learning-model-deployment-types-you-should-know-e7577251959) or more.

![Four Machine Learning Model Deployment Types](<Images/Types of Deployment.webp>)

### Batch
Runs regularly (e.g. every 10 minutes, 1 hour, 1 day, 1 week, etc.)

This is often used for marketing, e.g. churn prediction and then do something with the results.

### Web services
Something that needs to be provided on request. E.g. Taxi Ride Duration Prediction Service. It needs to be up and running all the time, but it only will make predictions when asked.

### Streaming
YOu have producers and consumers. Producers create events, and consumers have to react to these events. Usually it's a one-to-many or a many-to-many relationship between producers and consumers.

E.g. If a ride starts, then a ride duration predictor will keep updating, there's also a tip predictor. 

An alternative one is content moderation. Where the event is a video upload, and the consumers would be copyright violations, NSFW content detector, violence detectors, etc. These could then get merged into a final decision model which decides whether to delete it or promote the video.

## 4.2 Deployment as a webservice with Flask and Docker
Here the aim is to deploy the model and vectoriser stored in the pickle file from Module 1 [see this notebook to create the file](../../../01-intro/duration-prediction.ipynb). In this branch I have stored the model [here](01-intro/models/lin_reg.bin). Actually it is in `./04-deployment/web-service/lin_reg.bin` too.

This step is essentially going to replicate [Module 5 of the Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/05-deployment) just very briefly.

Essentially the steps are...
1. Create a virtual environment with pipenv
2. Put the model in a script
3. Put everything in a docker

### 4.2.1 Creating the virtual environment
#### Find out the exact versions of your libraries, resources, etc.
To find out which exact dependencies you are using you can just use.

```
$pip freeze 
```

However, this will return all the dependencies present which may be a lot so we can call specific parts using `grep` (**G**lobal **R**egular **E**xpression **P**rint) on the output. 

For those new to command line `|` is called a "pipe" and instructs the computer to take the output and perform the next command on it. While [grep](https://www.ibm.com/docs/en/aix/7.2?topic=g-grep-command) is a command that searches for a pattern in a file

```
$pip freeze | grep scikit-learn
```
So in the command above, we take the output of pip freeze and search for the regular expression `scikit-learn`. From this you should identify the version of all your key libraries etc.

For the example deployment we will need
* scikit-learn == 1.0.2
* flask
* python == 3.9 (For python 3.9 it is stable, a little higher, e.g. 3.10+ scikit learn may not work fully.)

#### Create and enter your venv
To create your environment with `pipenv install`

```
$pipenv install scikit-learn==1.0.2 flask --python=3.9
```

Then you can enter your venv with 
```
$pipenv shell
```
NB If you are using any form of conda this might not play so nicely with pipenv and so you might have a long prompt (or primary prompt string). To shorten it you can use
```
PS1="> "
```
This will truncate the prompt string dow to `> ` or whatever you set it to.

You should now have a `Pipfile` and a `Pipfile.lock` in your current directory.

To add further files you can just type 
```
$pipenv install YourLibrary
$pipenv lock #records the new requirements to the Pipfile.lock
```

To remove libraries, you can either
1. Manually delete them from the `Pipfile` then run
    ```
    $pipenv update #updates the Pipfile.lock
    $pipenv clean #removes the installed dependencies
    ```
2. Use the command uninstall. e.g.
    ```
    $pipenv uninstall numpy
    ```

### 4.2.2 Create predict.py
This is a python file that will be used to house the code for loading the model, and running the predictions

e.g. 
```
import pickle
from flask import Flask, request, jsonify

with open('./lin_reg.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'] + _ + ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features, model):
    X = dv.transform(features)
    preds = model.predict(X)
    return preds[0]

### Create wrapper for flask ###
app = Flask('duration-prediction)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride_info = request.get_json()

    features = prepare_features(ride_info)
    preds = predict(features)

    result = {
        'duration': preds
    }

    return jsonify(result)

if __name__.py == "__main__":
    app.run(debug=True, host='0.0.0.0, port=9696)

```
Then you can run the predict app locally with
```bash
$python predict.py
```



You can test out the predict function with a test.py script

```python
import requests

ride_info = {
    "PULocationID" : 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

url = 'http://127.0.0.1:9696/predict' 
requests.post(url, json=ride_info)
```
NB We use quotes here so it can be a `.json` in the future.

### 4.2.3 Run on a production WSGI server (Gunicorn)
Flask is run at the moment on a development server, whereas gunicorn can be used in production. Therefore to deploy a model as a web service it is recommended to use gunicorn as it can handle more traffic. There are more differences then this, I found an [article](https://stackshare.io/stackups/flask-vs-gunicorn#:~:text=Concurrency%20Model%3A%20Flask%20runs%20in,to%20handle%20multiple%20requests%20concurrently.) that lists the differences but in short they differ in...
* **Deployment Approach**: Flask is a micro web framework, and is useful for small apps and/or prototypes. Gunicorn is able to handle heavy traffic and high-loads.
* **Concurrency Model**: Flask can only handle one request at a time. Gunicorn uses a "pre-fork worker model" so it can handle multiple requests at the same time.
* **Scalability**: Flask doesn't scale easily with high volumes.
* **Load-balancing**: Flask doesn't have a built-in load balancer, so particular resources can become overloaded easily.
* **HTTP Server Features**: Gunicorn has more advanced HTTP functionality.
* **Memory Usage**: Gunicorn uses a lot more memory than flask.

There are also other options such as [Django](https://www.djangoproject.com/) to use instead of Flask for slightly larger apps. And you can also consider [Nginx](https://nginx.org/en/) which would be good for static stuff. However, they can run together too to take advantage of each other's strengths, [for example](https://vsupalov.com/gunicorn-and-nginx/). In general it seems that gunicorn is best suited to handle python web apps though.

to run Flask though gunicorn as long as gunicorn is installed just type...

```docker
$ gunicorn --bind=0.0.0.0:9696 predict:app
```
#### Pipenv --dev
NB The `requests` library is needed for development only so what we need to do is to create a dev environment that contains the `requests` library as well as what is needed for production.

```bash
pipenv install --dev requests
```

### 4.2.4 Package the app to the docker container
#### Create the dockerfile
So create the dockerfile to build the docker image

```docker
FROM python:3.9-slim #use the slim version to be more efficient
RUN pip install -U pip #just make sure the pip version is correct
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy #skips creating a venv aas were are already in a docker

COPY ["predict.py", "lin_reg.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
```
#### Build the image
```bash
$ docker build -t ride-duration-prediction-service:v1 .
```

#### Test/run the image
```bash
$ docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1
```
`-it`: Interactive mode, so cntrl + c will exit.
`--rm`: remove the docker file when finished
`-p`: Port mapping for host:container

### 4.2.5 What's upcoming
Integrating a model registry, so we don't need to be sharing models across e-mails pen drives etc. We can use the mlflow model

## 4.3 Web-services: Getting the models from the model registry

In 4.2 we just had a `model.bin` file which we used to deploy.

For 4.3 all the key files will be in `04-deployment/web-service-mlflow` where the model is stored in an S3 bucket (mlflow-models-alexy), while mlflow is run locally.

### Open up the MLflow server
In the cli type
```
mlflow server --backend-store-uri=sqlite:///mlflow.db --default-artifact-rot=s3://mlflow-models-alexey
```

### Run your experiments
The example is in `random-forest.ipynb` but the key code block is below this has the answers
```python
with mlflow.start_run():
    params = dict(max_depth=20, n_estimators=100, min_samples_leaf=10, random_state=0)
    mlflow.log_params(params)

    pipeline = make_pipeline(
        DictVectorizer(),
        RandomForestRegressor(**params, n_jobs=-1)
    )

    pipeline.fit(dict_train, y_train)
    y_pred = pipeline.predict(dict_val)

    rmse = mean_squared_error(y_pred, y_val, squared=False)
    print(params, rmse)
    mlflow.log_metric('rmse', rmse)

    mlflow.sklearn.log_model(pipeline, artifact_path="model")
```
### Load in the selected model from the artifacts database
By opening up either the `Client` or using the MLFlow UI you can identify which run you want or filter by aliases and/or tags to determine the best one to use.

In this example case the selected run is...
`RUN_ID = 'b4d3bca8aa8e46a6b8257fe4541b1136'`

So now we need to extract that model from our S3 bucket.
Given that we are working from the command line we can add the `RUN_ID` as an argument for the predict function.

Given that we have a dictionary vectoriser and the model. We need to add the mlflow and MlflowClient libraries too
```
import mlflow
from mlflow.tracking import MlflowClient
```
And also add a lot of bulky code in order to save the model and dictionary vectoriser to a temporary location etc.

A faster way would be to save a pipeline that contains the dictionary vecotriser and the model. I.e. this bit is added to your experiment and the whole pipeline be saved.

```python
from sklearn import make_pipeline

pipeline = make_pipeline(
    DictVectorizer()
    RandomForrestRegressor(**params, n_jobs=-1)
)
```

Then we can alter our `predict.py` code
```python
import os
import pickle

import mlflow
from flask import Flask, request, jsonify

MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

RUN_ID = 'e1efc53e9bd149078b0c12aeaa635df'
logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    #No need for dictionary vectorizer!#
    preds = model.predict(features)
    return float(preds[0])

app = Flask('duration-prediction')

@app-route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
```
NB you can also change the the way to load the model based on the stage, or alias, or s3 bucket, etc. See the [full documentation for pyfunc.load_model](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html?highlight=pyfunc%20load_model#mlflow.pyfunc.load_model) for all the details but I will list the key ones below

**For stage (to be depreciated eventually)**
```python
model = mlflow.pyfunc.load_model(f"models://{name}/{stage}")
```

**For aliases**
```python
alias = champion
champ_model = mlflow.pyfunc.load_model(f"models:/{name}@{alias}")
```

**For S3/google cloud storage bucket**
```python
#For S3
mlflow.set_tracking_uri("s3://<bucket-name>") #only when creating model
model = mlflow.pyfunc.load_model(s3://my_bucket/path/to/model)

#For GoogleCloud Storage
mlflow.set_tracking_uri("gs://<bucket-name>/mlflow") #only when creating model
model = mlflow.pyfunc.load_model(gs://my_bucket/path/to/model)
```

NB Alexey also recommends storing the run name so you can find out exactly which one it is from the registered model database.

### Becoming independent from the tracking server
Currently the code is dependent on being able to access the tracking server. However there could be issues with the server and it may go down for any number of reasons (e.g. Heavy Traffic). Therefore we may want to be able to use our model independently of the server.

In this case we can save our model(s) to a cloud location to use. And if it is google cloud storage just switch the `s3` to `gs`.

So this means that our logged model is now...
```
logged_model = f's3://mlflow-models-alexey/1/{RUN_ID}/artifacts/model'
```
NB In your deployment you can deploy with the `RUN_ID` as an environmental variable too.

## 4.4 (Optional) Streaming: Deploying models with Kinesis and Lambda
Kinesis and Lamda are AWS services. Kinesis is for streaming events and calling lambda.

**What are lambda functions?**
Lambda functions are serverless computing services, i.e. you can run code without provisioning or managing servers. Typically the code will run in response to events or HTTP requests, it automatically scales up and down resources as needed. 
* Lambda functions support: Node.js, Python, Java, C#, Go, Ruby, plus a few others
* Can be triggered by AWS services such as [S3 buckets](https://aws.amazon.com/s3/), [DynamoDB](https://aws.amazon.com/dynamodb/), [API Gateway](https://aws.amazon.com/api-gateway/), or even custom events
* Lambda functions work on a pay-per-use pricing model. So you pay based on the number of requests and the compute time consumed. *So be careful when debugging, and also exposing the address*.

The equivalent applications in GCP and Azure are [Google Cloud Functions](https://cloud.google.com/functions?hl=en) and [Azure Functions](https://azure.microsoft.com/en-us/products/functions).

**What is Kinesis?**
AWS Kinesis is a fully managed service that is used for real-time data streaming and analytics. There are four main parts
* **Kinesis Data Streams**: This is the core component that allows you to ingest and process large streams of data records in real time. It orders, buffers, and persists data streams. This helps you build apps that process data from various sources such as website clickstreams, IOT sensors, and social media feeds.
* **Kinesis Data Firehose**: This is a service that allows you to capture, transform, and load streaming data into AWS data stores such as [S3 buckets](https://aws.amazon.com/s3/), [Redshift](https://aws.amazon.com/redshift/) (Uses SQL to analyse structured and sem-structured data at scale), [Amazon OpenSearchService](https://aws.amazon.com/what-is/opensearch/) (I believe this helps you search for, and process data), etc.
* **Kinesis Data Analytics**: Allows you to run SQL or Java code on streaming data sources to perform analytics and gain insights in real-time.
* **Kinesis Video Streams**: Securely streams video from connected devices to AWS for analytics, ML, and/or other processing.

GCP and Azure equivalents are [Cloud Dataflow](https://cloud.google.com/dataflow?hl=en) and [Stream Analytics](https://azure.microsoft.com/en-us/products/stream-analytics) respectively. In GCP there's [Cloud Pub/Sub](https://cloud.google.com/pubsub/?hl=en) which is similar to Kinesis Data Streams. [Azure Event Hubs](https://learn.microsoft.com/en-us/azure/event-hubs/event-hubs-about) would be similar to Kinesis Data Streams too.

### When and when not to use Lambda and Kinesis Data Streams
#### When to use Lambda and Data Streams
A combination of Lambda and Kinesis works really well when you need to work on a per-record basis. You can still recieve batches but all the transformation will happen individually. See the image below.

![Best Practives for Consuming Amazon Kinesis data](Images/Best-practices-for-consuming-Amazon-Kinesis-1-1.jpg)
Image taken from taken from [*Best Practives for Consuming Amazon Kinesis data*](https://aws.amazon.com/blogs/big-data/best-practices-for-consuming-amazon-kinesis-data-streams-using-aws-lambda/)

#### When not to use Lambda and Data Streams
Each lambda function takes in individual records, or groups (known as *shards*) and is emphemeral i.e. it doesn't persist. This makes it hard for lambda functions to work if...
1. The data is either spread across multiple shards.
2. The data is "stateful" i.e. there are real-time updates to the data structure that require different lambda functions.
3.  You need to buffer large amounts of data.

According to [Best practices for consuming Amazon Kinesis Data Streams using AWS Lambda](https://aws.amazon.com/blogs/big-data/best-practices-for-consuming-amazon-kinesis-data-streams-using-aws-lambda/) for the first two cases [Kinesis Data Analytics](https://docs.aws.amazon.com/kinesisanalytics/latest/dev/what-is.html) should be used instead of Data Streams, while in the third instance [AWS Kinesis Firehose](https://aws.amazon.com/firehose/) should be used instead.

### Two ways to maps an AWS Lambda Consumer
AWS Lambda can map in two different ways to consume/use data records; standard iterators, and [enhanced fan-out (EFO) consumers](https://docs.aws.amazon.com/streams/latest/dev/enhanced-consumers.html). Standard ones are generally cheaper, and have more limits. While EFOs can handle more data faster, and send it to more places, but at a cost.

* **Standard iterators**: Lambda polls each shard in the stream once per second with through a HTTP protocol. When more records are available than it manage at once it will keep processing in batches until it can catch up with the stream again. The standard iterators also share the read-throughput with other consumers of the stream. There is a 2MB/sec output limit on each shard. You can increase this by utilising more, but smaller shards but the limit is set to five reads per second per [shard](https://docs.aws.amazon.com/streams/latest/dev/service-sizes-and-limits.html)
* **Enhanced Fan Outs**: To get around this you can create EFOs. Each EFO will recieve its own data stream, whose records will persist for longer, and can be delivered faster (<70 milliseconds). NB There is a limit of 20 EFOs per stream.

### Monitoring your Consumers
AWS recommends to use standard consumers when you have 3 or fewer consumers on a data stream, and low latency is not critical. This is because EFOs run at an additional cost per EFO consumer and per GB of data retrieved. NB These charges can be monitored with [AWS Cloud Watch](https://aws.amazon.com/cloudwatch/). There are a host of different aspects you need to consider which I believe will be addressed in Module 5. 

### Aim of this module
Our aim is to take the model that we built in 4.3 and get it to run in AWS.
i.e Pull the data and the model from the S3 bucket and pretend it is like a stream of data.

This is based off the [AWS Tutorial: Using Amazon Lambda with Amazon Kinesis](https://docs.aws.amazon.com/lambda/latest/dg/with-kinesis-example.html)

### How to set it up

#### Requirements

You can check your AWS CLI with ...
```bash
aws --version
```
You should see something greater than the values below
```bash
aws-cli/2.13.27 Python/3.11.6 Linux/4.14.328-248.540.amzn2.x86_64 exe/x86_64.amzn.2
```
If not you can [update the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) with one of the commands below.

**For Linux**

If this is the first time installing or updating you need to remove the pre-installed yum version first
```bash
sudo yum remove awscli
```
Then you can update with either...

*Linux x86 (64-bit)*
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```
Or Linux ARM
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```
**For Mac**
```bash
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
```
*NB For long commands, an escape character `\` is used to split a command over multiple lines.*

**For Windows**

Either download and run the [AWS CLI MSI installer](https://awscli.amazonaws.com/AWSCLIV2.msi) or in a terminal type
```
msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi
```

In my case I'm running
```
aws-cli/2.16.11 Python/3.11.8 Darwin/22.3.0 exe/x86_64
```

#### Create the execution role
Once you've logged into your IAMs AWS account (NB not the administrator one)
Navigate to the IAMs dashboard and create a role

**Settings**
* Trusted entity type: `AWS service`
* Use case: `Lambda`
![Role Settings](Images/CreateRoleSettings.png)

*Add permissions*
Permissions policies are a set of permissions that have already been pre-defined. You can select the pre-defined role `AWSLambdaKinesisExecutionRole` by clicking on the small "+" icon to the side of the role you can see the permissions granted. Specifically the `AWSLambdaKinesisExecutionRole` has the permissions that allow the lambda function to read items from Kinesis and write logs to CloudWatch logs.

![How to find pre-set permissions](Images/AWS_LambdaKinesisPermissions.gif)

Scroll down and click next to Name, review, and create the role.

Just assign the name as `lambda-kinesis-role` you can leave the rest of the entries in as default. So scroll down to the bottom and select the option "Create role". Once done you will be re-directed to the main roles page and you should see your role appearing. In the screen shot below it is in the bottom. 
![alt text](Images/RoleCreated_Success.png)

Also remember to attach the permissions to the role afterward too. If it didn't do this click on the `lambda-kinesis-role` and on the role's permissions pane you should be able to attach the required policies.

![How to attach policies](Images/Roles_AttachPolicies.png)

#### Create the function
1. Create a lambda function by navigating to the lambda fucntion setting. This can be easily done through the search bar. Or through the "Recently visted" pane, or the "All Services" option on the left hand side menu.

![alt text](Images/GoToLambda.png) 

Then once in the lambda function section select "Create Function" (big orange button)

**Settings**
* `Author from Scratch`

*Basic Information*
* Function name: `ride-duration-prediction-test`
* Runtime: `Python 3.9`
* Architecture: `x86_64`
* Permissions: `lambda-kinesis-role` NB here click on "Change default execution role" to select a pre-existing role"

Then scroll down to create function. (No need to alter any of the advanced settings for now). You will then be navigated to to the page for your new function.

![Lambda main page](Images/LambdaFunctionCreationSuccess.png)

#### Test the Lambda function
On the code source tab you are now able to go in and edit the code. Specifically the `lambda function`

The `lambda handler` is pre-configured to have two inputs
* `event`: This is the data that triggers the function. The lambda function will use this even to process the data. It can be in various forms.
    * For API Gateway events, it will contain details about the HTTP request e.g. headers, query parameters, body, etc.
    * For S3 events, it will contain details about the S3 object that triggered the event
    * For DynamoDB events, it will contain details about the DynamoDB data operations
* `context`: This is an object that provides methods and properties with information about the invocation, function, and execution environment. It is an optional argument and mainly used for logging or accessing details about the Lambda execution environment. Some example properties are...
    * `function_name`: The name of the Lambda function
    * `memory_limit_in_mb`: The configured memory limit
    * `aws_request_id`:  The unique request ID
    * `log_group_name`: The CloudWatch log group name
    * `log_stream_name`: The CloudWatch log stream name
    * `get_remaining_time_in_millis()` method to get the remaining milliseconds before the lambda execution times out

In our case we are going to be sending a JSON so in our test case we want to be able to see that the JSON is recieved correctly, and also that we generate a 200 status code *(If you want to understand status codes the [wiki](https://en.wikipedia.org/wiki/List_of_HTTP_status_codes#:~:text=1xx%20informational%20response%20%E2%80%93%20the%20request,syntax%20or%20cannot%20be%20fulfilled) article is actually pretty good.)*. 

Therefore we initially edit the code as follows...

```python
import json

def lambda_handler(event, context):
    print(json.dumps(event)) #Converts the JSON input into a dictionary
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }

```
Now we can deploy the function to make the changes live then we can configure the test event.

![How to configure test event](Images/Lambda_ConfigureTestEvent.png)

In the pop up modal our settings are as follows...

**Settings**

* *Test event action*: `Create new event`
* *Event name*: `test`
* *Event sharing settings*: `Private`
* *Event JSON*: 
    ```python
    {
        "key1": "value1",
        "key2": "value2",
        "key3": "value3"
    }
    ```
Now save it, and the modal should close. And when you click Test the output JSON should appear.

![Basic Lambda Test Function](Images/AWS_LambdaTestFunction.gif)

In reality our test function will be more complex, but it is recommended to structure it so you can test each step sequentially.

So now the lambda function is created, we now have to be able to send kinesis stream events to it. There at this testing phase it is recommended to keep the `print(json.dumps(event))` code there are as a means to check if we are recieving the stream correctly.

NB I showed how to do this with the UI, but it can also be done from the CLI with the following steps

1. Create a directory for the project and cd to that directory
    ```bash
    mkdir kinesis-tutorial
    cd kinesis-tutorial
    ```
2. Write/save your lambda function into the directory
3. Create a deployment package (aka zip it)
    ```bash
    zip function.zip <filename>
    ```
4. Create a Lambda function with the `create-function` command.
    ```bash
    aws lambda create-function --function-name ProcessKinesisRecords \
    --zip-file fileb://function.zip --handler index.handler --runtime nodejs18.x \
    --role arn:aws:iam::111122223333:role/lambda-kinesis-role
    ```
    NB Make sure you have got the `lambda function arn`
5. You can invoke a test with
    ```bash
    aws lambda invoke --function-name ProcessKinesisRecords \
    --cli-binary-format raw-in-base64-out \
    --payload file://input.txt outputfile.txt
    ```
    NB The `cli-binary-format` option is required if you're using AWS CLI version 2. To make this the default setting, run `aws configure set cli-binary-format raw-in-base64-out`.

#### Create a Kinesis stream
You can easily create a Kinesis stream with AWS CLI
```bash
aws kinesis create-stream --stream-name lambda-stream --shard-count 1
```
But the UI is better for when you are learning, so that's what I'll do here. Lets first navigate to the Kinesis Landing Page, it should look similar to the screenshot below.
![Kinesis Landing Page](Images/KinesisLandingPage.png)

Select `Kinesis Data Streams` and click `Create data stream`. NB Make sure you're in the same AWS region to save on prices etc.

**Settings**
* *Data stream name*: `ride_events`
* *Data stream capacity*:
    * *Capacity mode*: `Provisioned`
    * *Provisioned shards*: 1
NB Set it low to minimise costs. Pricing details are [here](https://aws.amazon.com/kinesis/data-streams/pricing/). Also remember to delete it afterwards.

After this just click 'Create data stream' at the bottom and you should be good to go.
![Successful creation of an AWS Kinesis Data Stream](Images/KinesisDataStream_SuccessCreation.png)

#### Add an event source in AWS Lambda
Now we need to connect the Kinesis stream to the Lambda function. To do this navigate back to the lambda function and add a trigger.

![Add Trigger](Images/Lambda_AddTrigger.png)

You'll then be taken to a page where we can specifcy the where the data inputs are coming from. In this case we want to select a `Kinesis stream` and from the drop down menu you should be able to find your `ride_events` stream.

![Trigger Configurations](Images/Lambda_TriggerConfigs.png)

#### Testing Kinesis uploads
Next we want to test to see if the stream is able to recieve. So we can just use the following lines in the CLI

```bash
KINESIS_STREAM_INPUT=ride_events
aws kinesis put-record \
    --stream-name ${KINESIS_STREAM_INPUT} \
    --partition-key 1 \
    --cli-binary-format raw-in-base64-out \
    --data "Hello, this is a test."
```
The `partition-key` flag instructs the kinesis stream on which shard to place the data event.

NB To run this code you need to have a [valid security token configured for AWS](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html?icmpid=docs_iam_console#Using_CreateAccessKey) 

Then configure AWS in your terminal e.g.
```bash
$ aws configure
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
Default region name [None]: us-west-2
Default output format [None]: json
```
Or you can use the [Cloud shell](https://aws.amazon.com/cloudshell/?nc2=h_ql_prod_dt_cs) too

Once you've entered the information into the terminal you should now be able to see it in the CloudLogs. THe cloud logs for the JSON are visible through the lambda function page. Go to the "Monitor" tab. Then select "View CloudWatch Logs, this will allow you to see all the events. By viewing the record we can use this as a test event which I configured as `test-kinesis` and is listed below.

```json
{
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": "1",
                "sequenceNumber": "49653169065253327118571418308087937388056434114127986690",
                "data": "SGVsbG8sIHRoaXMgaXMgYSB0ZXN0",
                "approximateArrivalTimestamp": 1718868621.288
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000000:49653169065253327118571418308087937388056434114127986690",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::094283992703:role/lambda-kinesis-role",
            "awsRegion": "us-east-1",
            "eventSourceARN": "arn:aws:kinesis:us-east-1:094283992703:stream/ride_events"
        }
    ]
}
```
Kinesis' `put-record` function should encode data to base64 prior to sending the data. This means that if we want to be able to read the data we will have to add in some additional code to our lambda function.

```python
import base64

for record in event['Records']:
    encoded_data = record['kinesis']['data'] #This accesses the actual data being sent
    decoded_data = base64.b64decode(encoded_data).decode('utf-8')
    print(decoded_data)
```
So now we can see if we deploy and then run the test it works and the text is decoded.
![Lambda Test - pt1](Images/Lambda-KinesisTest_1.png)

#### Testing Kinesis on the real data format
Now we want to test this on real data, 
```bash
aws kinesis put-record \
    --stream-name ${KINESIS_STREAM_INPUT} \
    --partition-key 1 \
    --cli-binary-format raw-in-base64-out \
    --data '{
        "ride": {
            "PULocationID": 130,
            "DOLocationID": 205,
            "trip_distance": 3.66}, 
        "ride_id": 156
        }'
```
NB From AWS CLIv2 onwards you need the additional tag `--cli-binary-format raw-in-base64-out` as put-record will assume a default entry of base64.

Now if you navigate to the CloudWatch logs of the most recent deployment (should be the top record). You should then see the results as below.

![Successfully reading in real data](Images/Lambda_RealDataTestSucess.png)

Finally, because we actually intend to pass this to another stream in we have ammended the function as below.

```python
def lambda_handler(event, context):
    
    #print(json.dumps(event)) #Converts the JSON input into a dictionary
    
    predictions = []
    for record in event['Records']:
        encoded_data = record['kinesis']['data'] #This accesses the actual data being sent
        decoded_data = base64.b64decode(encoded_data).decode('utf-8')
        ride_event = json.loads(decoded_data)
    
        ride = ride_event['ride']
        ride_id = ride_event['ride_id'] #This is so the data can be tracked in both the in and the out streams
        
        features = prepare_features(ride)
        prediction = predict(features)
        
        prediction_event = {
            'ride_duration': prediction,
            'ride_id': ride_id
            }
            
        predictions.append(prediction_event)
    
    return {
        'predictions': predictions
    }

```


#### Creating the outflow
##### Kinesis outflow/destination
Now we need to create a kinesis stream to record the outputs of the lambda function and store them

This can be done in the same way as the `ride_events` kinesis stream.
**Settings**
* *Data stream name*: `ride_predictions`
* *Data stream capacity*:
    * *Capacity mode*: `Provisioned`
    * *Provisioned shards*: 1

#### Changes to the Lambda function
NB Because this is streaming data we will need to add a whole load more data to the outputs so that they can be identified easily. e.g. Customer ids, model names, versions, etc. This keeps the data better organised and easier to identify in the future. We can reconfigure the outputs to reflect this, for now I have set them as strings.

##### Introducing Boto3
This will require [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) to send the data to the next kinesis stream. To cut a long story very short, Boto3 is used to allow python to interact with AWS services. Boto3 can be instatiated by
```python
kinesis_client = boto3.client('kinesis')
```

And each record can be added with the function `put_record` [docs](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/kinesis/client/put_record.html#put-record). NB there is also `put_records` but this is a little more expensive for smaller data. For larger batches it is cheaper than `put_record`.

For put record we have the following parts of data, in our situation we will use `StreamName`, `Data`, and `PartitionKey`.
```python
response = client.put_record(
    StreamName='string', #Can make it configurable with os.getenv
    Data=b'bytes', #Should be the predictions we want to send
    PartitionKey='string', #the identifier
    ExplicitHashKey='string', #Optional
    SequenceNumberForOrdering='string', #Optional
    StreamARN='string' #Optional
)
```
These changes can be seen in the current lambda function below

##### Current Lambda function
```python
import os
import json
import base64
import boto3

kinesis_client = boto3.client('kinesis')
PREDICTIONS_STREAM_NAME = os.getenv('PREDICTIONS_STREAM_NAME','ride_prediction') #Default stream name

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    return 10.0


def lambda_handler(event, context):
    
    #print(json.dumps(event)) #Converts the JSON input into a dictionary
    
    predictions = []
    for record in event['Records']:
        encoded_data = record['kinesis']['data'] #This accesses the actual data being sent
        decoded_data = base64.b64decode(encoded_data).decode('utf-8')
        ride_event = json.loads(decoded_data)
    
        ride = ride_event['ride']
        ride_id = ride_event['ride_id'] #This is so the data can be tracked in both the in and the out streams
        
        features = prepare_features(ride)
        prediction = predict(features)
        
        prediction_event = {
            'model': 'ride_duration_prediction_model',
            'version': '123',
            'prediction': {
                'ride_duration': prediction,
                'ride_id': ride_id   
            }
        }
        # Add to Kinesis Client
        kinesis_client.put_record(
            StreamName=PREDICTIONS_STREAM_NAME,
            Data=json.dumps(prediction_event),
            PartitionKey=str(ride_id)
        )    
        predictions.append(prediction_event)
    #This output is just for us not the stream
    return {
        'predictions': predictions
    }
```
#### Updating Permissions
Currently our IAMS policy only has permission to read, but not to write to a kinesis stream.

So we need to create a new policy and add it to the role.
1. Navigate to the policies page, and select 'Create policy' on the top right
    ![alt text](Images/Policy_Create.png)
2. Set up the policy
    
    **Settings**
    * *Service*: `Kinesis`
    * *Actions allowed*: Write-> `PutRecord` and `PutRecords`
    * *Resources*: Specify the Stream ARN. If you specify the ARN then the rest of the data should autofill
    ![Specifying ARN](Images/Policy_SpecifyARN.png)
3. Click next to go to "Review and Create"
    **Settings**
    * *Policy name*: `lambda_kinesis_write_to_ride_predictions`
    * *Description* : `Allows the function/user/role to write records to the ride_predictions stream`

    Now select create policy
4. Attach policy to the role. You can do this by navigating back to the roles section and then just add the policy you've created just like before
![Attached policy](Images/Roles_CustomerPolicyAttached.png)
5. Test now you should be able to deploy, if there is an error check for typos. e.g. I typed in `ride_prediction` instead of `ride_predictions` for the stream name so it didn't work.

#### Reading from the stream
You can navigate to your terminal and read from the stream with the following command.
```bash
KINESIS_STREAM_OUTPUT='ride_predictions'
SHARD='shardId-000000000000'

SHARD_ITERATOR=$(aws kinesis \
    get-shard-iterator \
        --shard-id ${SHARD} \
        --shard-iterator-type TRIM_HORIZON \
        --stream-name ${KINESIS_STREAM_OUTPUT} \
        --query 'ShardIterator' \
)

RESULT=$(aws kinesis get-records --shard-iterator $SHARD_ITERATOR)

echo ${RESULT} | jq -r '.Records[0].Data' | base64 --decode
``` 

NB [jq](https://ioflood.com/blog/jq-linux-command/#:~:text=The%20jq%20command%20in%20Linux%20is%20a%20versatile%20tool%20that,simplify%20working%20with%20JSON%20data.) is a tool that allows you to parse and manipulate JSON data from your command line. It's linux based, but you can install it on Mac with `brew install jq` and on windows with `winget install jqlang.jq`.

#### Adding the model
So far, we've been able to read in the Kinesis stream and output the predictions into another stream. However, to make a valid prediction rather that just returning "10.0" we need to load in the model and to do this we need to execute the lambda function within a docker.

So first using VS Code (or another IDE) you can copy and paste your lambda function across and save it as `lambda_function.py`. NB For ease you can see the final version from Alexey [here](https://github.com/mleiwe/mlops-zoomcamp/tree/main/04-deployment/streaming).

The main task is uploading the model from our S3 bucket and then using the prediction.

So in the lambda code we can load in the model from our MLflow s3 bucket with the following code snippet...
```python
import mlflow
RUN_ID = os.getenv('RUN_ID')

logged_model = f's3://mlflow-models-alexey/1/{RUN_ID}/artifacts/model'
# logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)
``` 
In this case the RUN_ID will be an environmental variable that can be set in our terminal.


The final predict function will therefore look like this.
```python
def predict(features):
    pred = model.predict(features)
    return float(pred[0])
```
NB At this point it is worth [testing](../../../04-deployment/streaming/test.py) the predict function so that we know it works locally before adding it to the online lambda function.

If you don't want to click the link, you can just copy and paste the code below.

```python
import lambda_function

event = {
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": "1",
                "sequenceNumber": "49630081666084879290581185630324770398608704880802529282",
                "data": "ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogMTMwLAogICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LAogICAgICAgICAgICAidHJpcF9kaXN0YW5jZSI6IDMuNjYKICAgICAgICB9LCAKICAgICAgICAicmlkZV9pZCI6IDI1NgogICAgfQ==",
                "approximateArrivalTimestamp": 1654161514.132
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000000:49630081666084879290581185630324770398608704880802529282",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::387546586013:role/lambda-kinesis-role",
            "awsRegion": "eu-west-1",
            "eventSourceARN": "arn:aws:kinesis:eu-west-1:387546586013:stream/ride_events"
        }
    ]
}
result = lambda_function.lambda_handler(event, None)
print(result)
```

Now we can test this in the terminal by navigating to the folder where the `lambda_function.py` and `test.py` are, and then executing the test.py with...

```bash
export PREDICTIONS_STREAM_NAME="ride_predictions"
export RUN_ID="e1efc53e9bd149078b0c12aeaa6365df"
export TEST_RUN="True"
python test.py
```

We created an additional argument `TEST_RUN` which only when false will we add data to the `ride_predictions` stream.

Plus, we don't have access to Alexey's S3 bucket with the model so I did a quick fix to make it work

Loading the model
```python
import pickle
model_path = '../../cohorts/2024/04-deployment/homework/model.bin'
with open(model_path, 'rb') as f_in:
    dv, model = pickle.load(f_in)
```
Change the predict function
```python
def predict(features):
    features = dv.transform(features)
    pred = model.predict(features)
    return float(pred[0])
```

The full code can be seen [here](04-deployment/streaming/lambda_function_NoS3.py) with an ammended test.py [here](04-deployment/streaming/test_NoS3.py)

#### Creating the docker
As documented earlier
1. Create the pipefile and pipefile.lock
```bash
pipenv install boto3 mlflow scikit-learn --python=3.9
```
2. Create the dockerfile
```docker
FROM public.ecr.aws/lambda/python:3.9

RUN pip install -U pip
RUN pip install pipenv 

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "lambda_function.py", "./" ]

CMD [ "lambda_function.lambda_handler" ] #This says where to find the lambda function --> similatr to GUNICORN
```
NB Make sure you have the correct base image, this needs to be configurable in AWS. There is the [Amazon ECR Public Gallery](gallery.ecr.aws) from where you can select from many base images. See the .gif below for an example for how to get the name of the image you want.

![Finding the AWS Lambda Docker Images](Images/FindDockerImage.gif)

3. Build the docker
```bash
docker build -t stream-model-duration:v1 .
```

4. Run the docker
Remember now you need to transfer your environmental variables too.
```bash
docker run -it --rm \
    -p 8080:8080
    -e PREDICTIONS_STREAM_NAME="ride_predictions" \
    -e RUN_ID="e1efc53e9bd149078b0c12aeaa6365df" \
    -e TEST_RUN="True" \
    stream-model-duration:v1
```
NB before configuring the final set we want to test the whole set up locally too. Th
```python
import requests 

event = {
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": "1",
                "sequenceNumber": "49630081666084879290581185630324770398608704880802529282",
                "data": "ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogMTMwLAogICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LAogICAgICAgICAgICAidHJpcF9kaXN0YW5jZSI6IDMuNjYKICAgICAgICB9LCAKICAgICAgICAicmlkZV9pZCI6IDI1NgogICAgfQ==",
                "approximateArrivalTimestamp": 1654161514.132
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000000:49630081666084879290581185630324770398608704880802529282",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::387546586013:role/lambda-kinesis-role",
            "awsRegion": "eu-west-1",
            "eventSourceARN": "arn:aws:kinesis:eu-west-1:387546586013:stream/ride_events"
        }
    ]
}


url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
response = requests.post(url, json=event)
print(response.json())
```

NB In this case in dockerfile the line `CMD [ "lambda_function.lambda_handler" ]` will expose the lambda function. And when we run the docker we are publishing the 8080 port with the flag `-p 8080:8080`. But the url can be a little opaque. Here's how it breaks down

* `http://localhost:8000`: This is the local address and port where the Lambda Runtime Interface Emulator (RIE) is listening. The port 9000 is commonly used, but it can be changed if needed.
* `/2015-03-31`: This is a version identifier for the Lambda invocation API. It's a fixed part of the URL and doesn't change.
* `/functions/function`: This path segment indicates that we're invoking a function. The word "function" here is a placeholder and doesn't need to be changed for different functions.
* `/invocations`: This final part of the path specifies that we want to invoke the function.

You will also have to specify the region in the `kinesis client` in boto3. This is not necessary for the lambda function.

The region is the Kwarg `region`, which could be hardcoded but it is probably better to set it as an environmental variable. Within AWS there are certain default environmental variables that will be present in all commands. The full list is [here](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-envvars.html). Therefore we just need to set these up when we run our docker.

The necessary ones will be `AWS_ACCESS_KEY_ID`,`AWS_SECRET_ACCESS_KEY`,`AWS_DEFAULT_REGION`. Possibly also your `AWS_SESSION_TOKEN` if you are using temporary tokens. This means our docker run command will look like the following.

```bash
docker run -it --rm \
    -p 8080:8080 \
    -e PREDICTIONS_STREAM_NAME="ride_predictions" \
    -e RUN_ID="e1efc53e9bd149078b0c12aeaa6365df" \
    -e TEST_RUN="True" \
    -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
    -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
    -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}" \
    stream-model-duration:v1
```
That could be seen as quite troublesome, so you can also set them in the .aws config folder

Alternatively, you can mount the `.aws` folder with your credentials to the `.aws` folder in the container:

```bash
docker run -it --rm \
    -p 8080:8080 \
    -e PREDICTIONS_STREAM_NAME="ride_predictions" \
    -e RUN_ID="e1efc53e9bd149078b0c12aeaa6365df" \
    -e TEST_RUN="True" \
    -v /Users/Marcus/.aws:/root/.aws \
    stream-model-duration:v1
```
### Publish your Docker to ECR
[Amazon Elastic Container Registry (ECR)](https://docs.aws.amazon.com/gb_en/AmazonECR/latest/APIReference/Welcome.html) is a fully managed Docker container registry service provided by AWS. Basically a little bit like a AWS specific GitHub for docker images. You can store docker images privately or publically too. NB there is a [price](https://aws.amazon.com/ecr/pricing/) if you go beyond the limits. Currently (June 2024) this is set at 500MB per month for private storage and 50GB per month of public storage. So you really have to store large images to hit this limit.

You can publish your images using the AWS CLI tool (there's no web UI I could find).
```bash
aws ecr create-repository --repository-name duration-model
```

There will be an output where there is are several fields in a json format
```json
{
    "repository": {
        "repositoryArn": "arn:aws:ecr:<region>:<registryId>:<repository>/duration-model",
        "registryId": "<registryId>",
        "repositoryName": "duration-model",
        "repositoryURI": "<registryId>.dkr.<region>.amazonaws.com/duration-model",
        "createdAt": "<TIMESTAMP>",
        "imageTagMutability": "MUTABLE",
        "imageScanningConfiguration": {
            "scanOnPush": false
        },
        "encryptionConfiguration": {
            "encryptionType": "AES256"
        }
    }
}
```
The key field here is the repository Uri field which we will use to push the local image to the repository. I recommend saving it as a environmental variable. We can then push the local image to the ECR. 

First you will need to sign in which can be done with your AWS credentials.
```bash
$(aws ecr get-login --no-include-email)
```
Now you can push the image
```bash
REMOTE_URI="387546586013.dkr.ecr.eu-west-1.amazonaws.com/duration-model"
REMOTE_TAG="v1"
REMOTE_IMAGE=${REMOTE_URI}:${REMOTE_TAG}

LOCAL_IMAGE="stream-model-duration:v1"
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE}
docker push ${REMOTE_IMAGE}
```
### Create the final lambda function

#### Initial Settings
So now back on the AWS page, navigate to the [Lambda functions page](https://us-east-1.console.aws.amazon.com/lambda/home?region=us-east-1#/functions), and select "Create Function". This time instead of "Author from scratch" select "Container image" instead.
![Build a Lambda function from a container](Images/Lambda_CreateFromECRDocker.png)

**Settings**

*Basic Information*
* `Function name`: "ride-duration-prediction"
* `Container image URI`: {The image URI}
* `Architecture`: x86_64

*Permissions*
* `Execution role`: Use an existing role
    * "lambda-kinesis-role"

The select "Create function" and you will be taken to the function's page.

#### Set configurations
What we now need to do is configure the function. This means choosing which triggers to use, what are the environmental variables etc. These can be found in the Configurations tab

##### Environmental Variables
The environmental variables you will need are...
* `PREDICTION_STREAM_NAME`
* `RUN_ID`
The AWS credentials should not be used as each person using the lambda function should use their own.

##### General configuration
Click the "Edit" button on the top right of this pane. Depending on your model requirements you may need to increase the memory. e.g. In this example
* `Memory`: 512 MB
* `Ephermeral storage`: 512
* `Timeout`: 0 min 30 sec

Then save. NB The more memory you allocate the more expensive an invocation is. You can check from a Test what are the memory usages.

#### Adding triggers
Now add the kinesis stream as the trigger. This is the `ride_events` stream we created earlier.

![Add Trigger](Images/Lambda_ConfigureTestEvent.png)

NB we should delete the existing test lambda function as we don't really want to have two lambda functions reading the same stream with the same purpose.

Now to test the lambda function we send an event to the `ride-events` stream.

```bash
aws kinesis put-record \
    --stream-name ${KINESIS_STREAM_INPUT} \
    --partition-key 1 \
    --data '{
        "ride": {
            "PULocationID": 130,
            "DOLocationID": 205,
            "trip_distance": 3.66
        }, 
        "ride_id": 156
    }'
```

#### Add S3 bucket permission policy
You will need to add a policy permission to list and read S3 buckets to the role for this to work. Being lazy just grant all read and list permissions.

Then Specify the resource, which is done in two places
* Specify the ARN of your S3 bucket: This will pop up in a self-explanatory modal.
* Specify the object ARN: Here provide the Bucket name, and set the object as `*` to read all objects in the bucket.

Now move to the review section and fill in the details as appropriate. e.g.
* `Name`: read_permission_mlflow_models
* `Description`: This allows you to read the MLflow models


### Calling/Invoking the Lambda function for non stream events
If you aren't using a stream you can send JSON requests to your functions in the following ways

1. Using AWS Lambda Function URLs:

    AWS Lambda Function URLs provide a built-in HTTPS endpoint for your Lambda function. Probably the simplest way to do it. The URL structure is as follows:

    `https://<url-id>.lambda-url.<region>.on.aws`
    
    * `url-id`: A unique identifier for the function URL.
    * `region`: The AWS region where the Lambda function is deployed. 
2. Using AWS SDK or CLI:

    When invoking a Lambda function using the AWS SDK or CLI, the URL structure is managed internally by the SDK or CLI. However, the general format for the API endpoint is:

    `https://lambda.<region>.amazonaws.com/2015-03-31/functions/<function-name>/invocations`

    * `region`: The AWS region where the Lambda function is deployed.
    * `function-name`: The name of the Lambda function.
3. Using API Gateway:

    When invoking a Lambda function through API Gateway, the URL structure is defined by the API Gateway configuration. The general format is:

    `https://<api-id>.execute-api.<region>.amazonaws.com/<stage>/<resource>`
    * `api-id`: The unique identifier for the API Gateway.
    * `region`: The AWS region where the API Gateway is deployed.
    * `stage`: The stage name (e.g., dev, prod).
    * `resource`: The resource path defined in the API Gateway.

4. Use the AWS Console:
    
    This is what we used when testing the function. You could in theory manually change the inputs each time. I wouldn't recommend this but you could do it if you really want.
    
#### Clean up your resources (deleting them)
The beauty of Lambda functions is that you don't pay for them if you aren't using them. However as mentioned, you will be billed for each kinesis stream you leave running. So it is best to delete them once you've finished. To do this just simply navigate to the Kinesis dashboard then select the streams you aren't running anymore and select delete from the options.

![Deleting you data stream](Images/KinesisDataStream_Delete.png)
 
### Useful resources
But I found these blog posts/documentations helpful
* [What is AWS Lambda](https://docs.aws.amazon.com/lambda/latest/dg/welcome.html)
* [Best practices for consuming Amazon Kinesis Data Streams using AWS Lambda](https://aws.amazon.com/blogs/big-data/best-practices-for-consuming-amazon-kinesis-data-streams-using-aws-lambda/)
* [How Lambda processes records from Amazon Kinesis Data Streams](https://docs.aws.amazon.com/lambda/latest/dg/with-kinesis.html#:~:text=You%20can%20use%20a%20Lambda,consumer%20with%20enhanced%20fan%2Dout.)
* [AWS Lambda function types](https://docs.aws.amazon.com/lambda/latest/dg/services-kinesis-create.html)
* [Enhanced Fan Output documentation](https://docs.aws.amazon.com/streams/latest/dev/enhanced-consumers.html)
* [Amazon Elastic Container Registry (ECR) basic information](https://docs.aws.amazon.com/gb_en/AmazonECR/latest/APIReference/Welcome.html)

## 4.5 Batch: Preparing a scoring script
To be done in the future.


