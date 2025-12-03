'''
This script contain al of the fuctions that are used for benchmarking
'''

def mask_filter(df, col, crit): # (DataFrame, Column name, Criteria)
    return df[df[col] > crit]

def query_filter(df, col, crit):
    return df.query(f'{col} > {crit}')
