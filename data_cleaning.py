import numpy as np
import pandas as pd
import re
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

sensor = pd.read_csv('csv/industrial_iot_sensor_data_repository.csv')

#drop nan in sensor data:
sensor.dropna(subset = ['timestamp', 'sensor_id', 'temperature', 'vibration', 'pressure', 'operational_status', 'energy_consumption', 'fault_flag', 'decision_label'], inplace=True)

#Convert timestamp to datetime:
sensor.timestamp = pd.to_datetime(sensor.timestamp, format='%d-%m-%y %H:%M')

#Normalize the columns ['operational_status', 'decision_label'] according to the dictionary:

sensor.operational_status = sensor.operational_status.str().strip().str.lower()
operational_dict = {'maintenance':'MNT', 'operational':'OPT', 'failure':'FAIL'}
sensor.operational_status = sensor.operational_status.map(operational_dict)

sensor.decision_label = sensor.decision_label.str.strip().str.lower()
decision_dict = {'failure':'FAIL', 'optimal':'OPT', 'degraded':'DEG'}
sensor.decision_label = sensor.decision_label.map(decision_map)

#Add ['year','month','day','hour'] columns based on timestamp:
for i in ['year','month','day','hour']:
    sensor[i] = getattr(sensor.timestamp.dt, i)

#flag high risk readings of vibration > 4.5:
sensor['vibration_risk'] = sensor.vibration > 4.5

#Create a pivot_table that is based on temperature and listing hours and operational status:
pivot = pd.pivot_table(sensor, values = 'temperature', index = 'hour', columns = 'operational_status', aggfunc = 'mean')

#Create plots for the pivot table:
sns.heatmap(pivot, cmap='coolwarm')
plt.title('Energy usage by status and hour')
plt.savefig('plots/sensor_temperature_by_time_and_status.png')
