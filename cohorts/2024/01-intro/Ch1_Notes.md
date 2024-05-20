# MLOps Introduction
Putting ML into production

## 1.1 - Introduction
MLOps = Set of best practices for putting ML into production

This course will be based on solving the issue of predicting the duration of the trip.

NB Very simplified process are three stages
1. Design: Is ML really the right solution?
2. Train: Find the best possible model 
3. Operate: Run/deploy the model, also evaluate and update the model .

## 1.2 Preparing the Venv
This is how to prepare on AWS.
1. Log on AWS
2. Create an EC2 instance
3. Launch the instance
4. Select OS
    * Recommendation is Ubuntu
    * 64-but (x86)
    * **NB This may cost money to run the EC2 instance depending on what you require. 16GB RAM is recommended which according to the video will cost money to run**

Extenstion Remote-ssh can be used so you can run VS Code from a remote server

*need docker, anaconda, jupyter notebooks etc. Try to run this*
This is an abridged version of the notes which the YT video for more detailed notes

Can also use Google Cloud Platform which is what I've done. detailed notes are [here](https://github.com/mleiwe/mlops-zoomcamp/blob/main/cohorts/2024/01-intro/GoogleCloudSetUpNotes.md)
 
### Clone the repo
If you want to push and pull then you need to configure the SSH passwords, however if you only want to pull then you can simply `git clone` the https version. Make sure you are in the folder where you want to pull the repo

    $git clone https://github.com/DataTalksClub/mlops-zoomcamp.git

### Install `Remote-SSH` to use VS Code in your machine
To run Jupyter Notebooks in a remote session via VS Code, you need to download the extension `Remote-SSH` which you can easily find by clicking on the extensions icon, and seaching for it, and downloading it if it is not already uploaded

Then all that is needed is to begin typing in the command pallet (cmd/cntrl + shift + p)

ALternatively on the bottom left of VS Code there is a small blue icon which you can use to connect your VM to VS code and operate everything from this VS Code window or a new one. 

### Set up jupyter notebooks
Within your virtual machine, navigate to `~/mlops-zoomcamp` and from there make a new directory called notebooks.

    $ mkdir notebooks

What we need to do now is **port forwarding** where we connect the remote port (i.e. your VM) to your local port.
1. `cmd/cntrl + ~` to open a terminal if one isn't available

2. Navigate to `Ports` and click on the big blue button saying "Forward a Port"
    1. The port we want to forward is `8888`, which is where Jupyter Notebooks is. 
![alt text](<Screenshot 2024-05-20 at 6.03.03 PM.png>)

3. Now in the VM terminal set up a jupyter notebook using the following command

    jupyter notebook

This will then ask if you want to open it in a browser (say yes) and it should appear

4. Password/token
This will now ask you to type in a password or token but the easiest thing to do is just copy and paste one of the URLs provided into your address bar and you are good to go.

## 1.3a Parquet  format
The data is now stored as parquet files instead of csv files. Parquet files are much smaller meaning they are more effective. You can still `wget` the files though. NB you will also need to `!pip install pyarrow` in order to read the files

## 1.3b How to train a model (Optional)
1. `!wget <files>` for training. The taxi cab data can be found [here](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

2. Load in your parquet files into a pandas dataframe
Once you have pyarrow installed this should be relatively easy just type

    df = pd.read_parquet(<path to file>)

into your notebook. This should be readable provided pandas is imported as pd.

3. Calculate your target variable (duration)
Unlike the csv files the parquet files also stores the data in the datetime format which makes it much easier to operate.

    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']

In this case `duration` is in timedelta format which is not easily trainable. We need to convert this to minutes. This can be done with using the `timedelta.total_seconds()` function then dividing by 60 to get minutes.

    df['duration_mins'] = df['duration'].apply(lambda td: td.total_seconds() / 60)

4. Filtering
First look at the distribution of the duration values using seaborn.

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.distplot(df_jan['duration'])

It should look something similar to the plot below.

![text](image.png)

Clearly we need some means of filtering the data. One good way is by removing the extreme percentiles at either end. We can determine the percentile cutoffs with the describe function to see where extreme values lie.

    df_jan['duration'].describe(percentiles=[0.01, 0.05, 0.95, 0.98, 0.99])

Which produces..

    count    3.066766e+06
    mean     1.566900e+01
    std      4.259435e+01
    min     -2.920000e+01
    1%       7.833333e-01
    5%       3.300000e+00
    50%      1.151667e+01
    95%      3.646667e+01
    98%      4.873333e+01
    99%      5.725000e+01
    max      1.002918e+04
    Name: duration, dtype: float64

Personally I would trim evenly based on percentages. But in this study they use all rides >= 1min and <= 60mins.

    df_jan_filt = df_jan[(df_jan['duration'] >=1) & (df_jan['duration']<=60)]


