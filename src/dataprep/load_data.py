"""" Unzip data zip file and export a parquet file to data folder.

The data for the project is stored as a zip file in the data directory
This script perform the data unloading steps. The zip file is unzipped, as a json file.
The json file is loaded to pandas Dataframe and then the data is export as a parquet file
for efficient storage.

functions:
    *unzip_file - unzip the zip file and export it
    *export_parquet - read in json file and export file as parquet
    *delete_json - deletes the exported json file
    *main - the main function of the script

"""
#import hydra and replace dir
import zipfile
from pathlib import Path
import pandas as pd


def unzip_file(file_path, export_path):
    """ unzip a zip file and export it at a given location

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

    try:
        with zipfile.ZipFile(file_path, "r") as data_zip:
            data_zip.extractall(export_path)
            print(f"Done extracting into json file {export_path}")

    except Exception as error:
        print(f"unzip_func -> An error occur while extracting file: {error}")


def export_parquet(file_path, export_path):
    """
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

    try:
        dataframe = pd.read_json(file_path, orient="index")
        dataframe.reset_index(inplace=True)
        dataframe.rename(columns={0: "Target", "index": "Log"}, inplace=True)
        dataframe.to_parquet(export_path, compression="gzip")
        print(f"Done Exporting parquet file to {export_path}")

    except Exception as error:
        print(f"An error occur while exporting file: {error}")

def delete_json(file_path):
    """ Delete the json file to free up space

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
            print(f"File {file_path} has been succesfully deleted.")

    except FileNotFoundError:
        pass

    except Exception as error:
        print(f"An error occur while trying to delete the file: {error}")


def main():
    """
    run script

    """
    project_dir = Path.cwd()
    zipfile_loc = Path(project_dir, "data/raw/convolve-epoch1.zip")
    raw_data_loc = Path(project_dir, "data/raw/")
    json_file_loc = Path(raw_data_loc, "train.json")
    csv_loc = Path(raw_data_loc, "raw_train.gzip")
    print(zipfile_loc)
    #main
    try:
        unzip_file(file_path= zipfile_loc, export_path= raw_data_loc)
        export_parquet(file_path= json_file_loc, export_path= csv_loc)
        delete_json(json_file_loc)

    except Exception as error:
        print(f"An error occur while trying to delete the file: {error}")
