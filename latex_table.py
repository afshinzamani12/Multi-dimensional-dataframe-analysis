'''
'latex_table.py'
This program is create LaTex tables from DataFrames
Usage:
    from latex import latex_table
    latex_table(df)
The output will be 'table.tex'
'''


def latex_table(df):
    column_format = ''.join(['>{\\raggedright\\arraybackslash}p{3cm}'] * len(df.columns))
    df.to_latex(
    'table.tex',
    index=True,
    column_format=column_format,
    escape=False,     # Important if your column names contain underscores
    longtable=False,  # Set True if table may span pages
    multicolumn=True, # Enable auto \multicolumn headers
    multirow=True # Enable for mullti-index
    )
