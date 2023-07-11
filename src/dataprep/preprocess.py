"""
doc
"""
# 4m sample
# Split data to train, valid, test using 80,10,10
# load data to tensor dataset, use prefech, shuffle, interleave, and preprocess


import string
import re
import polars as pl
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
nltk.download('stopwords')



def load_spit(file, class_label):
    """doc"""
    datarame = pl.read_parquet(file)

    train_data, data = train_test_split(
        datarame, random_state=42, test_size=0.2, stratify=datarame[class_label]
    )
    valid_data, test_data = train_test_split(
        data, random_state=42, test_size=0.5, stratify=data[class_label]
    )
    new_data = (train_data, valid_data, test_data)
    return new_data

def clean_text_pro(text):
    """doc
    """
    text = text.lower()
    text = re.sub('<[^>]*>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = [word for word in text.split()
            if word not in stop_words and word not in string.punctuation]
    text = ' '.join(word for word in text)
    return text

def encoder(label):
    """doc
    """
    if label == 'normal':
        label = 0
    else:
        label = 1
    return label

def save_to_parquet(dataframe, file_loc):
    """doc
    """
    dataframe.write_parquet(file =file_loc, compression = 'gzip')


def mani():
    dataframes_set = load_spit()
    for dataset in dataframes_set:
        dataset = dataset.with_columns(pl.col('Target').apply(encoder, return_dtype=pl.Int32))
        dataset = dataset.with_columns(pl.col('Log').apply(clean_text_pro))
        save_to_parquet(dataset,)
    
if __name__ == '__main__':
    pass