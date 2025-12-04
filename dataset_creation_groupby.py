import numpy as np
import pandas as pd
from function_lib import exp, linear, log_n, trig
func_map = {'S1':exp, 'S2':linear, 'S3':log_n, 'S4':trig}
trial = ['First','Second','Third','Fourth']

def dataframe_groupby(arr, index, columns, p='Temperature'):
    df = pd.DataFrame(arr, index = index, columns = columns)
    df_r = df.reset_index()
    df_r['group_id'] = df_r.groupby([df_r['time'].dt.date, df_r['Sensor'], df_r['Location']]).ngroup()
    import time
    start = time.time()
    for col in trial:
        df_r[(p, col)] = df_r.groupby('group_id').apply(lambda group: pd.Series(func_map[group['Sensor'].iloc[0]](np.arange(len(group)), p, seed=int(group.name)), index=group.index)).reset_index(level=[0,1], drop=True)
    df_t = time.time() - start # Element 0 of the array
    df_r.drop(columns=('group_id',''), inplace=True);
    df = df_r.set_index(['Location','Sensor','time'])
    return df, df_t
