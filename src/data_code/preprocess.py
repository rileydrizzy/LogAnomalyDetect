
"""
doc
"""
# 4m sample
#Split data to train, valid, test using 80,10,10
#load data to tensor dataset, use prefech, shuffle, interleave, and preprocess

from pathlib import Path

import polars as pl
from sklearn.model_selection import train_test_split

def load_spit(file, class_label):
    """doc
    """
    datarame = pl.read_parquet(file)
    train_data, data = train_test_split(datarame,random_state= 42, test_size= 0.2
                                  ,stratify= class_label)
    valid_data, test_data = train_test_split(data, random_state= 42, test_size = 0.5
                                             ,stratify= class_label)
    return train_data, valid_data, test_data

def clean_text_pro():
    """doc
    """




class LabelEncoder:
    """doc
    """
    def __init__(self):
        self.label_mapping = {}
        self.inverse_mapping = {}

    def fit(self, labels):
        """doc
        """
        unique_labels = set(labels)
        self.label_mapping = {label: index for index, label in enumerate(unique_labels)}
        self.inverse_mapping = {index: label for label, index in self.label_mapping.items()}

    def transform(self, labels):
        """doc
        """
        return [self.label_mapping[label] for label in labels]
    
    def fit_transform(self, labels):
        """doc
        """
        self.fit(labels)
        result = self.transform(labels)
        return result
    def inverse_transform(self, encoded_labels):
        """doc
        """
        return [self.inverse_mapping[label] for label in encoded_labels]
