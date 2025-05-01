import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


PATH = "./Cancer_Data.csv"

# def pandas_to_numpy(df):
#     df = pd.DataFrame(df)
#     return df.to_numpy()


def process(path):
    df = pd.read_csv(path)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    print("\nDimensions of dataset (rows, columns):")
    print(df.shape)

    # check and drop null values
    print("\nCheck for null values:")
    print(df.isnull().sum())

    # check and drop NaN values
    print("\nCheck for NaN values:")
    print(df.isna().sum())
    df = df.dropna(axis=1, how='all') # Dropping column #32 that is a column of NaNs

    df = df.drop(columns=["id"]) # Redundant ID column

    # df.value_counts()

    # labels column
    # 1 = malignant, 0 = benign
    df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]
    true = df["diagnosis"].to_numpy()
    print("\nHead of labels array:")
    print(true[:5])

    # select variables from dataset
    features = ['symmetry_mean', 'texture_mean', 'smoothness_mean', 'concave points_mean', 'fractal_dimension_mean', 'radius_mean']
    input = df.drop(["diagnosis"], axis=1)
    input = df.filter(features)
    print("\nHead of input array:")
    print(input.head())

    # normalize variables
    input_normalized = (input - np.min(input)) / (np.max(input) - np.min(input))
    input_normalized = input_normalized.to_numpy()

    # split data into training and testing sets
    # 80% training, 20% testing
    input_bias = np.hstack([np.ones((input_normalized.shape[0], 1)), input_normalized])
    input_train, input_test, true_train, true_test = train_test_split(input_bias, true, test_size=0.2, random_state=42)


    return input_train, input_test, true_train, true_test

if __name__ == "__main__":
    process(PATH)
