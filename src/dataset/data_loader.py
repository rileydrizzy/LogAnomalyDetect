""""
This script perform the data unloading steps. The zip file is unzipped, as a json file.
The json file is loaded to pandas Dataframe and then the data is export as a parquet file
for efficient storage.
Data(parquet file) is read and splitted into train, valid and test datasets using a 80:10:10 ratio. 
The data was splitted using stratfied strategy to account for the class imbalanced
and then each dataset was then save as a parquet for efficient storage.

functions:
    *unzip_file - unzip the zip file and export it
    *export_parquet - read in json file and export file as parquet
    *delete_json - deletes the unused json file, to clean storage
    *load_split - load and split the dataset into train, valid and test set
    *save_to_parquet - save and export dataset as parquet
    *main - the main function to run the script

"""
import zipfile
from pathlib import Path

import hydra
import pandas as pd
import polars as pl
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from utils.logger import logger


def unzip_file(file_path, save_path):
    """unzip a zip file and export it at a given location

    Parameters
    ----------
    file_path : str, Path object
        The file path of the zip file
    export_path : str
        The path where to export the unzip the file

    Returns
    -------
        None

    """

    with zipfile.ZipFile(file_path, "r") as data_zip:
        data_zip.extractall(save_path)


def export_parquet(file_path, save_path):
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
    dataframe.to_parquet(save_path, compression="gzip")


def delete_json(file_path):
    """Delete the unused json file to free up storage space

    Parameters
    ----------
    file_path : str, Path object
        The file path of the json file

    Returns
    -------
        None
    """
    file_path = Path(file_path)
    try:
        if file_path.exists() and file_path.is_file():
            file_path.unlink()

    except FileNotFoundError:
        pass


def load_spit(file_path, target):
    """create dataframe and split the data into train, valid and test set,
    in a 80:10:10 ratio.
    The data will be splitted using stratified strategy to account for the class imbalanced

    Parameters
    ----------
    file_path : str, Path object
        The file path of the file
    target : str
        The column name of the target

    Returns
    -------
    new_data : tuple
        A tuple containing the train_data, valid_data, test_data in that order
    """

    datarame = pl.read_parquet(file_path)

    train_data, data = train_test_split(
        datarame, random_state=42, test_size=0.2, stratify=datarame[target]
    )
    valid_data, test_data = train_test_split(
        data, random_state=42, test_size=0.5, stratify=data[target]
    )
    datasets_tuple = (train_data, valid_data, test_data)
    return datasets_tuple


def save_to_parquet(dataframe, save_path):
    """save the dataframe as a parquet file

    Parameters
    ----------
    dataframe : dataframe
        A dataframe
    file_path : str. Path object
        The path where to export the parquet

    Returns
    -------
        None
    """
    dataframe.write_parquet(file=save_path, compression="gzip")
    return


@hydra.main(config_name="config", config_path="config", version_base="1.2")
def main(cfg: DictConfig):
    """
    run script

    """
    logger.info("Commencing the data unzipping process.")
    try:
        unzip_file(file_path=cfg.files.raw_data, save_path=cfg.paths.data_raw)
        export_parquet(file_path=cfg.files.json_file, save_path=cfg.files.parquet_file)
        delete_json(file_path=cfg.files.json_file)

        logger.success("Data has been unzipped and saved at {cfg.files.parquet_file}")
        logger.info("Initiating the data splitting procedure.")

        dataframes_set = load_spit(file_path=cfg.files.parquet_file, target="Target")
        dataframes_paths = (
            cfg.files.train_dataset,
            cfg.files.valid_dataset,
            cfg.files.test_dataset,
        )
        for index, dataset in enumerate(dataframes_set):
            save_to_parquet(dataframe=dataset, save_path=dataframes_paths[index])
        logger.success("Data has been splitted and saved at {cfg.paths.data_processed}")

    except Exception:
        logger.exception("Data unloading was unsuccesfully")


if __name__ == "__main__":
    main()
