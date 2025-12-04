# This script is created based on the project outlined in "Multidim_sensor_analysis.tex" file. The output results are also included in the same latex file.

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import re

rand = np.random.RandomState(42)

# Set and create output folders for plots and latex tables:
import os
os.makedirs("output/plots", exist_ok=True)
os.makedirs("output/tables", exist_ok=True)

location = ['North','South','East','West']
sensor = ['S1','S2','S3','S4']
#Create arrays for columns indexes:
trial = ['First','Second','Third','Fourth']
columns_temp = pd.MultiIndex.from_product([['Temperature'], trial])
columns_humid = pd.MultiIndex.from_product([['Humidity'], trial])

from pre_setup_dataframes import zeros_arr

# Set the dataframe numbers all to zero initially:
start_date = '2012-01-01'
end_date = '2018-06-01'
Temp_am, index_am = zeros_arr(start_date, end_date, time_part='am')
Temp_pm, index_pm = zeros_arr(start_date, end_date, time_part='pm')
Humidity_am, index_am = zeros_arr(start_date, end_date, time_part='am')
Humidity_pm, index_pm = zeros_arr(start_date, end_date, time_part='pm')

'''
This section is for creation of controlled random data using defined functions in the file function_library:
'''
# Create dataframes by groupby method and apply functions:
from function_lib import exp, linear, log_n, trig
func_map = {'S1':exp, 'S2': linear, 'S3': log_n, 'S4':trig}
# Regenerate dataframes (with zero data):
df1 = pd.DataFrame(Temp_am, index=index_am, columns=columns_temp)
df2 = pd.DataFrame(Temp_pm, index=index_pm, columns=columns_temp)
df3 = pd.DataFrame(Humidity_am, index=index_am, columns=columns_humid)
df4 = pd.DataFrame(Humidity_pm, index=index_pm, columns=columns_humid)
df1_resetted = df1.reset_index()
df2_resetted = df2.reset_index()
df3_resetted = df3.reset_index()
df4_resetted = df4.reset_index()

df1_t, df2_t, df3_t, df4_t = [[0] for _ in range(4)] # dataframes creation times (Saved as an arrays)
df_dict = {'df1':[df1_resetted, 'Temperature', df1_t],
           'df2':[df2_resetted, 'Temperature', df2_t],
           'df3':[df3_resetted, 'Humidity', df3_t],
           'df4':[df4_resetted, 'Humidity', df4_t]}
for i, value in enumerate(df_dict.values()):
    value[0]['group_id'] = value[0].groupby([value[0]['time'].dt.date, value[0]['Sensor'], value[0]['Location']]).ngroup()
    import time
    start = time.time()
    for col in ['First','Second','Third','Fourth']:
        value[0][(value[1], col)] = value[0].groupby(value[0]['group_id']).apply(lambda group: pd.Series(func_map[group['Sensor'].iloc[0]](np.arange(len(group)), value[1], seed=int(group.name)), index=group.index)).reset_index(level=[0,1], drop=True)
    value[2][0] = time.time() - start # Element 0 of the array
    value[0].drop(columns=('group_id',''), inplace=True);
    print(f'df{i+1} was created successfully. Creation time: {value[2][0]}')

df1 = df1_resetted.set_index(['Location','Sensor','time'])
df2 = df2_resetted.set_index(['Location','Sensor','time'])
df3 = df3_resetted.set_index(['Location','Sensor','time'])
df4 = df4_resetted.set_index(['Location','Sensor','time'])

# Merge and concatnation of the dataframes (df1, df2, df3, df4) into a single df
def merge_dataframes(df1, df2, df3, df4):
    df_13 = pd.merge(df1, df3, left_index=True, right_index=True) # Add two dataframes horizontally with shared indexes
    df_24 = pd.merge(df2, df4, left_index=True, right_index=True)
    df = pd.concat([df_13, df_24], axis=0) # Add the previous dataframes vertically
    return df

df = merge_dataframes(df1, df2, df3, df4)

#This is sample indexing of the multidim dataframes:
idx = pd.IndexSlice
df1.loc[idx['North', 'S1', '2012'], idx['Temperature', 'First']]
df1.head()

# Add column of "Average" to each dataframe by averaging trials of each row:
df1.loc[:, idx['Temperature', 'Average']]=df1.mean(axis=1)
df2.loc[:, idx['Temperature', 'Average']]=df2.mean(axis=1)
df3.loc[:, idx['Humidity', 'Average']] = df3.mean(axis=1)
df4.loc[:, idx['Humidity', 'Average']] = df4.mean(axis=1)

# Add a column of Flag based on the average values of temperature and humidity to the dataframes:
Temp_bins = [10, 22.5, 35, 47.5, 60]
Humidity_bins = [0, 25, 50, 75, 100]
labels = ['Low', 'Below average','Above average','High']
# Here we use pd.cut to add labels to the the last columns of dataframes:
df1.loc[:, idx['Temperature', 'Condition']] = pd.cut(df1.loc[:, idx['Temperature', 'Average']], bins=Temp_bins, labels=labels, include_lowest=True, right=True)
df2.loc[:, idx['Temperature', 'Condition']] = pd.cut(df2.loc[:, idx['Temperature', 'Average']], bins=Temp_bins, labels=labels, include_lowest=True, right=True)
df3.loc[:, idx['Humidity', 'Condition']] = pd.cut(df3.loc[:, idx['Humidity', 'Average']], bins=Humidity_bins, labels=labels, include_lowest=True, right=True)
df4.loc[:, idx['Humidity', 'Condition']] = pd.cut(df4.loc[:, idx['Humidity', 'Average']], bins=Humidity_bins, labels=labels, include_lowest=True, right=True)

# Add operator based on day (and no hourly) from a random dictionary:
operators = ['Goerge','Michael','David','Sara','Alison','John']
unique_dates_am = pd.Series(df1.index.get_level_values('time').normalize().astype(str)) # Extract the unique dates from the dataframes
unique_dates_pm = pd.Series(df3.index.get_level_values('time').normalize().astype(str))
operators_map_am = {date: rand.choice(operators) for date in unique_dates_am}
operators_map_pm = {date: rand.choice(operators) for date in unique_dates_pm}
# Only add 'Operator' column to the df3 and df4 (humidity dataframes) as they are going to appear on the right hand side of the merged df and they share the same operator with temperature.
df3.loc[:, idx['Operator']] = df3.index.get_level_values('time').normalize().astype(str).map(operators_map_am)
df4.loc[:, idx['Operator']] = df4.index.get_level_values('time').normalize().astype(str).map(operators_map_pm)

# Apply irregularity to the 'operator' column of all dataframes
# The function is inside the file "string_irregularity.py" in the current directory.
from string_irregularity import add_irregularities
df3.loc[:, idx['Operator']] = df3.loc[:, idx['Operator']].apply(add_irregularities)
df4.loc[:, idx['Operator']] = df4.loc[:, idx['Operator']].apply(add_irregularities)
df = merge_dataframes(df1, df2, df3, df4)

# Use RegEx for cleaning messy data of operators name:
df.loc[:, idx['Operator']] = df.loc[:,idx['Operator']].str.findall(r'[A-Za-z]+').apply(lambda x: ''.join(x)).str.capitalize()
# Check if all of the values in 'Operator' column of the df is equal to the operators dictionary:
operators_mixed_order = df.Operator.unique()
if set(operators) == set(operators_mixed_order):
    print('Cleaning of the "operators" column performed successfully')

# The average and std of Temperture and Humidity measured by each operator:
print(f'Averge and std of temperature measured by operator are as below:\n',
df.loc[:, idx['Temperature', 'Average']].groupby(df.Operator).agg(['mean', 'std']))
print(f'Averge and std of humidity measured by operator are as below:\n',
df.loc[:, idx['Humidity', 'Average']].groupby(df.Operator).agg(['mean', 'std']))

# The average and std by year for temperature and humidity:
time_index_year = df.index.get_level_values(2).year
print(f'Average and std of temperature by year:',
      df.loc[:,idx['Temperature', 'Average']].groupby(time_index_year).agg(['mean','std']))
print(f'Average and std of humidity by year:\n',
      df.loc[:,idx['Humidity', 'Average']].groupby(time_index_year).agg(['mean','std']))

# Create pivot tables:
df_resetted = df.reset_index() # Set index as columns
#print(df_resetted.columns) # Prints the column names (tuples): Useful in pivot_table
df_resetted[('year','')] = df_resetted[('time','')].dt.year # Add year column based on timestamp.
import time
pivot = {'Sensor':[None, None], 'Location':[None, None], 'Operator':[None, None]} #First value is for dataframe and second value is benchmarking time of the pivot creation
fig, ax = plt.subplots(len(pivot), sharex=True, figsize=(8, 12))
for i, (key, value) in enumerate(pivot.items()):
    start = time.time()
    value[0] = df_resetted.pivot_table(values=[('Temperature','Average')], index=[('year','')], columns=[(key,'')], aggfunc=['mean', 'std'])
    value[1] = time.time()-start
    value[0]['mean'].plot(ax=ax[i])
    ax[i].set_title(f'pivot by {key}')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.savefig('output/plots/Temp_avg_year.pdf')

# Create the same tables as pivot using grouping and aggregation method:
group = {'Sensor':[None, None], 'Location':[None, None], 'Operator':[None, None]}
for (key,value) in group.items():
    start = time.time()
    value[0] = df_resetted.groupby([('year',''),(key,'')])[[('Temperature','Average')]].agg(['mean','std']).unstack()
    value[1] = time.time()-start

# Bar graphs of the pivot tables:
fig, ax = plt.subplots(len(pivot), sharex=True, figsize=(8, 14))
for i, (key,value) in enumerate(pivot.items()):
    x = np.arange(len(value[0].index))
    width = 1/(len(value[0].columns)/2+1)
    list1 = value[0].columns.get_level_values(-1).unique().tolist()
    for j, S in enumerate(list1):
        x_shift = (2*j-(len(value[0].columns)/2-1))*width/2
        ax[i].bar(x-x_shift, value[0][('mean','Temperature','Average',S)], yerr=value[0][('std','Temperature','Average',S)], width=width, label=S, capsize=3)
    ax[i].set_xticks(x)
    ax[i].set_xticklabels(value[0].index)
    plt.tight_layout()
    ax[i].legend()
    ax[i].set_ylabel('Average temperature (C)')

plt.xlabel('year')
plt.savefig('output/plots/Temp_avg_year_bar.pdf')

# Masking via query() and eval() and benchmarking performance
df_flattened = df.copy()
df_flattened.columns = ['_'.join(col) for col in df_flattened.columns]
df_flattened.columns
print('The masking time is:')
start = time.time()
df_flattened[(df_flattened['Temperature_Average']>35)&(df_flattened['Humidity_Average']>50)]
print(time.time()-start)
print('The query time is:')
start = time.time()
df_flattened.query('Temperature_Average > 35 and Humidity_Average>50')
print(time.time()-start)

# Benchmarking:
from benchmark import benchmark
from benchmarking_functions import mask_filter
from benchmarking_functions import query_filter

#This is for creating latex tables out of the four timing dictionaries:
index = pd.MultiIndex.from_product([['Masking', 'Query'], ['Temp', 'Humid']], names=['Category','Measurement'])
columns = ['First','Second','Third','Fourth']
data_temp = np.zeros((4,4))
df_mask_query = pd.DataFrame(data_temp, index=index, columns=columns)

df_mask_query.iloc[0, :] = [benchmark(lambda: mask_filter(df_flattened, df_flattened.columns[i], 35)) for i in range(4)]
df_mask_query.iloc[1, :] = [benchmark(lambda: mask_filter(df_flattened, df_flattened.columns[i], 50)) for i in range(7,11)]
df_mask_query.iloc[2,:] = [benchmark(lambda:query_filter(df_flattened, df_flattened.columns[i], 35)) for i in range(4)]
df_mask_query.iloc[3,:] = [benchmark(lambda: query_filter(df_flattened, df_flattened.columns[i], 50)) for i in range(7,11)]

df_pivot_group = pd.DataFrame(np.zeros((2,3)), index=['pivot','groupby'], columns=['First','Second','Third'])
for i, (value1, value2) in enumerate(zip(pivot.values(), group.values())):
    df_pivot_group.iloc[0,i] = value1[1]
    df_pivot_group.iloc[1,i] = value2[1]

# Create a benchmark table comparing loop and groupby methods for creating dataframes:
from dataset_creation_by_loop import dataframe_loop
from dataset_creation_groupby import dataframe_groupby
start_date = '2012-01-01'
end_dates = ['2012-04-01 23:00', '2012-07-01 23:00', '2013-07-01 23:00', '2015-01-01 23:00', '2018-01-01 23:00']
# Remember to include hour (23:00) for the dates. Otherwise it will end up with some empty cells in the dataframes.
df_loop_t = np.zeros(len(end_dates))
df_t = np.zeros(len(end_dates))
for i, date in enumerate(end_dates):
    zeros_df, index = zeros_arr(start_date, date, 'am')
    df_loop, df_loop_t[i] = dataframe_loop(zeros_df, index, columns_temp, p='Temperature')
    df, df_t[i] = dataframe_groupby(zeros_df, index, columns_temp, p='Temperature')
# Bechmark latex table:
timing_table = pd.DataFrame(np.zeros((2, len(end_dates))), index=['Looping method', 'Groupby method'], columns=end_dates)
for i in range(len(end_dates)):
    timing_table.iloc[0,i] = df_loop_t[i]
    timing_table.iloc[1,i] = df_t[i]

# This section is for exporting latex tables (Add table names into the dictionary before running):
dict_latex_tables = {'df_mask_query.tex':df_mask_query, 'df_pivot_group.tex': df_pivot_group, 'dataframe_creation_time.tex': timing_table}

for (key, value) in dict_latex_tables.items():
    value.to_latex(f'output/tables/{key}', multirow=True, escape=False)
    pass
else:
    print('Exporting latex tables were performed successfully')
# Compile latex:
import subprocess
def compile_tex(filename="multidim_sensor_analysis.tex"):
    subprocess.run(['pdflatex', filename], check=True)
if __name__=="__main__":
    compile_tex()
    print("Latex compilation finished!")

# ************** End of Program ****************
print('This program ended successfully')    
