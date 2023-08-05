import hydra
import pandas as pd
from omegaconf import DictConfig


def export_parquet(file_path, export_path):
    """read in json file, create and adjust data which headers.
    then save and export data as parquet for efficient storage

    Parameters
    ----------
    file_path : str, Path object
        The file path of the json file
    export_path : str
        The path where to export the parquet file

    Returns
    -------
        None
    """
    dataframe = pd.read_json(file_path, orient="index")
    dataframe.reset_index(inplace=True)
    dataframe.rename(columns={0: "Target", "index": "Log"}, inplace=True)
    dataframe.to_parquet(export_path, compression="gzip")


json_file = "/workspaces/log_anomaly/data/raw/train.json"
paq = "/workspaces/log_anomaly/data/raw/raw_parquet"

df = pd.read_json(json_file, orient="index")
print(df.shape)
