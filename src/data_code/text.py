""""
doc
"""
import zipfile
from pathlib import Path

import pandas as pd

project_dir = Path.cwd()
zipfile_loc = Path(project_dir, "data/raw/convolve-epoch1.zip")
raw_data_loc = Path(project_dir, "data/raw/")
json_file_loc = Path(raw_data_loc, "train.json")


def unzip_file(zip_path, export_path):
    """
    doc
    """
    with zipfile.ZipFile(zip_path, "r") as data_zip:
        data_zip.extractall(export_path)
    print(f"Done extracting into json file {export_path}")


def csv_loader(json_file, export_path):
    """
    doc
    """
    dataframe = pd.read_json(json_file, orient="index")
    dataframe.reset_index(inplace=True)
    dataframe.rename(columns={0: "Target", "index": "Log"}, inplace=True)
    dataframe.to_parquet(export_path, compression="gzip")
    print(f"Done Exporting parquet file to {export_path}")


def main():
    """
    doc
    """
    unzip_file(zipfile_loc, raw_data_loc)
    csv_loc = Path(raw_data_loc, "raw_train.gzip")
    csv_loader(json_file_loc, csv_loc)


if __name__ == "__main__":
    main()
