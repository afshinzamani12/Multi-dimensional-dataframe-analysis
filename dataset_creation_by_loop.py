import numpy as np
import pandas as pd
from function_lib import exp, linear, log_n, trig
func_map = {'S1':exp, 'S2':linear, 'S3':log_n, 'S4':trig}

def dataframe_loop(arr, index, columns, p = 'Temperature'):
    df_loop = pd.DataFrame(arr, index = index, columns = columns)
    # Reset index for the created dataframe (Required for groupby method)
    df_loop_r = df_loop.reset_index()
    # Add "group_id" columns to the resetted dataframes above:
    df_loop_r['group_id'] = df_loop_r.groupby([df_loop_r['time'].dt.date,df_loop_r['Sensor'], df_loop_r['Location']]).ngroup()
    import time
    start = time.time()
    for col in range(4,7):
        for i in range(int(len(df_loop_r)/12)):
            df_loop_r.iloc[12*i:12*(i+1),col] = func_map[df_loop_r['Sensor'][12*i]](np.arange(12), p, seed=df_loop_r['group_id'][12*i])
    df_loop_t = time.time() - start
    df_loop_r.drop(columns=('group_id',''), inplace=True)
    df_loop = df_loop_r.set_index(['Location','Sensor','time'])
    return df_loop, df_loop_t

