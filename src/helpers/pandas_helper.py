from functools import reduce
import pandas as pd

def filter_dict(df, dic):
    return df[df[dic.keys()].apply(
            lambda x: x.equals(pd.Series(dic.values(), index=x.index, name=x.name)), asix=1)]