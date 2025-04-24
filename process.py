import pandas as pd
import numpy as np


def pandas_to_numpy(df):
    df = pd.DataFrame(df)
    return df.to_numpy()