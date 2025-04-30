import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


PATH = "./Cancer_Data.csv"

def pandas_to_numpy(df):
    df = pd.DataFrame(df)
    return df.to_numpy()


def process(path):
    df = pd.read_csv(path)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    df = df.dropna(axis=1, how='all') # Dropping column #32 that is a column of NaNs
    df = df.drop(columns=["id"]) # Redundant ID column
    df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]
    df.isnull().sum()
    df.value_counts()
    true = df["diagnosis"].to_numpy()
    input = df.drop(["diagnosis"], axis=1)
    input_normalized = (input - np.min(input)) / (np.max(input) - np.min(input))
    input_train, input_test, true_train, true_test = train_test_split(input_normalized, true, test_size=0.2, random_state=42)


    return input_train, input_test, true_train, true_test

if __name__ == "__main__":
    process(PATH)