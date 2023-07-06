""""
doc
"""
from pathlib import Path
import zipfile
import pandas as pd

project_dir = Path.cwd().parent.parent
data_zip_loc = Path(project_dir, "data/raw/convolve-epoch1.zip")
raw_data_loc = Path(project_dir,"data/raw/")
csv = Path(raw_data_loc, "train.csv")
json_ = Path(raw_data_loc, "train.json")

def unzip_file(filename, zipath):
    """
    doc
    """
    with zipfile.ZipFile(filename,  'r') as data_zip:
        data_zip.extractall(zipath)
    print(f"Done extracting into {zipath}")


def csv_loader(train_path):
    """
    doc
    """
    dataframe = pd.read_json(train_path, orient='index')
    dataframe.reset_index(inplace = True)
    dataframe.rename(columns= {0 : 'Target', 'index' :'Log'}, inplace= True)
    dataframe.to_csv(csv)
    print(f"Done Exporting to {train_path}")

def main():
    """
    doc
    """
    unzip_file(data_zip_loc,raw_data_loc)
    csv_loader(json_)

if __name__ == "__main__":
    main()
