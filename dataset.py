import kagglehub
from kagglehub import KaggleDatasetAdapter


def getSet():
    # Download latest version
    path = kagglehub.dataset_download("erdemtaha/cancer-data")

    print("Path to dataset files:", path)

# Don't use; can manipulate via pandas
def loadData():
    # Set the path to the file you'd like to load
    file_path = ""

    # Load the latest version
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "erdemtaha/cancer-data",
        file_path,
        # Provide any additional arguments like
        # sql_query or pandas_kwargs. See the
        # documenation for more information:
        # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )

