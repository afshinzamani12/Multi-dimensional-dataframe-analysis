# This function is to create the dataset with all data equal to zero.

import numpy as np
import pandas as pd

def create_empty_datasets(start_date, end_date):
    time = pd.date_range(start_date, end_date, freq='h')
    time_am = time[time.hour < 12]
    time_pm = time[time.hour >= 12]
index_am = pd.MultiIndex.from_product([location, sensor, time_am], names=['Location','Sensor','time'])
index_pm = pd.MultiIndex.from_product([location, sensor, time_pm], names=['Location','Sensor','time'])

#Create arrays for columns indexes:
trial = ['First','Second','Third','Fourth']
data_type = ['Temperature','Humidity']
columns_temp = pd.MultiIndex.from_product([['Temperature'], trial])
columns_humid = pd.MultiIndex.from_product([['Humidity'], trial])

rand = np.random.RandomState(42)
# Set the dataframe numbers all to zero initially:
Temp_am = np.zeros((len(location)*len(sensor)*len(time_am), len(trial)))
Temp_pm = np.zeros((len(location)*len(sensor)*len(time_pm), len(trial)))
Humidity_am = np.zeros((len(location)*len(sensor)*len(time_am), len(trial)))
Humidity_pm = np.zeros((len(location)*len(sensor)*len(time_pm), len(trial)))
