import pandas as pd
import numpy as np


def pandas_to_numpy(df):
    df = pandas.DataFrame(df)
    return df.to_numpy()