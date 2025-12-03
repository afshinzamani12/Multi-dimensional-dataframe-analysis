# This script is an excercise that covers Ufunctions, index alignment and slicing, masking , ...

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt

rand = np.random.RandomState(42)

day1_df = pd.DataFrame(rand.randint(20,50,(24,4)), columns = ['S1','S2','S3','S4'])

day2_df = pd.DataFrame(rand.randint(20,50,(24,4)), columns = ['S1','S2','S3','S4'])

day3_df = pd.DataFrame(rand.randint(20,50,(24,4)), columns = ['S1','S2','S3','S4'])

#Define index:
day1_df.index = [f"2025-09-01 {i}:00" for i in range(len(day1_df))]
day2_df.index = [f"2025-09-02 {i}:00" for i in range(len(day1_df))]
day3_df.index = [f"2025-09-03 {i}:00" for i in range(len(day1_df))]

data = pd.concat([day1_df, day2_df, day3_df], axis = 0)

#Create a Series "baseline" with values [22.0, 21.5, 22.5, 23.0] and matching the sensor names:

baseline = pd.Series(np.array([22.0, 21.5, 22.5, 23.0]), index = ['S1', 'S2', 'S3', 'S4'])

#subtract baseline from data:

data_reduced = data.sub(baseline)

#Drop one column and then subtract the baseline (If one column ia missing:

data_dropped = data.drop(columns = "S4")
data_dropped - baseline
#OR
data.sub(baseline, axis=1)

#This will add NaN column in the fourth column.

#Apply np.abs to the reduced dataframe:
np.abs(data_reduced)

#Normalised data:
normalized_data = (data - data.mean())/data.std()

#Slice all readings from '2025-09-02 0:00' to '2025-09-02 12:00':
day2_first_half = data.loc['2025-09-02 0:00':'2025-09-02 12:00', :]

#Slice every third row starting from second row:
data_every_third = data.iloc[2::3,:]

#slice columns 'S2' and 'S4' using fancy indexing:
data.loc[:,['S2','S4']]

#mask all readings where drift exceeds 2.5 C:
data[data[(data - data.mean())>2.5].notna().all(axis = 1)]
drift = data - data.mean()

#Mask rows where any sensor has a reading below 40 C:
data[data[data<40].notna().all(axis = 1)]

#Create a mask for rows where 'S1' and 'S3' both exceed 25 C:
data[(data.S1 > 25) & (data.S2 > 25)]

#Add rolling data average columns to DataFrame:
rolling_mean = {}
rolling_mean_df = pd.DataFrame(rolling_mean)
for i in range(1,5):
    rolling_mean_df[f'S{i}'] = drift[f'S{i}'].rolling(window=3).mean()

#Mask the rolling mean where it exceeds 1.5 C and visualise the results:
masked_rolling = rolling_mean_df.mask(rolling_mean_df>1.5)

#drop all rows where all sensors have NaN after masking:
rolling_mean_df.mask(rolling_mean_df>1.5).dropna(how='all')

#plot the results of masked rolling:
plt.figure(figsize=(12, 6))
masked_rolling.plot(ax=plt.gca(), marker='o', linestyle='-', alpha=0.8)

plt.title("Masked Rolling Mean of Sensor Drift (Window = 3)", fontsize=14)
plt.xlabel("Timestamp")
plt.ylabel("Drift (°C)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="Sensors")
plt.tight_layout()
plt.savefig('plots/df_masked_rolling.png')
