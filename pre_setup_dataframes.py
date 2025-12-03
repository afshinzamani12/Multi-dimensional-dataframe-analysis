import numpy as np
import pandas as pd

location = ['North','South','East','West']
sensor = ['S1','S2','S3','S4']
trial = ['First','Second','Third','Fourth']
def zeros_arr(start_date, end_date, time_part='am'):
    #Create arrays for indexes:
    time = pd.date_range(start_date, end_date, freq='h')
    if time_part == 'am':
        Time = time[time.hour < 12]
    elif time_part == 'pm':
        Time = time[time.hour >= 12]
    
    row_index = pd.MultiIndex.from_product([location, sensor, Time], names=['Location','Sensor','time'])
    zeros_array = np.zeros((len(location)*len(sensor)*len(Time), len(trial)))
    return zeros_array, row_index
