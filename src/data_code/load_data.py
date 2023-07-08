"""" Unzip data zip file and export a parquet file to data folder.

The data for the project is stored as a zip file in the data directory
This script perform the data unloading steps. The zip file is unzipped, as a json file.
The json file is loaded to pandas Dataframe and then the data is export as a parquet file
for efficient storage.
For the Script to run correctly, it should be run on the project directory

functions:
    *unzip_file - unzip the zip file, exporting to the specifity loacation
    and prints message of success
    *csv_loader - read in json file, export file as parquet and prints message 
    *del_loader - deletes the exported json file and prints done message
    *main - the main function of the script

"""

import zipfile
from pathlib import Path

import pandas as pd


def unzip_file(zip_path, export_path):
    """unzip s

    Parameters
    ----------
    file_loc : str
        Str
    print_cols : bool, optional
        A flag used to print the columns to the console (default is
        False)

    Returns
    -------
        None

    Raises
    ------
        RuntimeError
            Amount Withdrawn greater than Total_Money
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as data_zip:
            data_zip.extractall(export_path)
            print(f"Done extracting into json file {export_path}")
    except Exception as error:
        print(f"An error occur while extracting file: {error}")


def csv_loader(json_file, export_path):
    """
    doc
    """
    try:
        dataframe = pd.read_json(json_file, orient="index")
        dataframe.reset_index(inplace=True)
        dataframe.rename(columns={0: "Target", "index": "Log"}, inplace=True)
        dataframe.to_parquet(export_path, compression="gzip")
        print(f"Done Exporting parquet file to {export_path}")

    except Exception as error:
        print(f"An error occur while exporting file: {error}")


def delete_json(json_file):
    """
    doc
    """
    file = Path(json_file)
    try:
        if file.exists() and file.is_file():
            file.unlink()
            print(f"File {json_file} has been succesfully deleted.")
    except FileNotFoundError:
        pass
    except Exception as error:
        print(f"An error occur while trying to delete the file: {error}")


def main():
    """
    doc
    """
    project_dir = Path.cwd()
    zipfile_loc = Path(project_dir, "data/raw/convolve-epoch1.zip")
    raw_data_loc = Path(project_dir, "data/raw/")
    json_file_loc = Path(raw_data_loc, "train.json")

    unzip_file(zipfile_loc, raw_data_loc)
    csv_loc = Path(raw_data_loc, "raw_train.gzip")
    csv_loader(json_file_loc, csv_loc)
    delete_json(json_file_loc)


if __name__ == "__main__":
    main()
