"""Test Module for Dataset Loader

This module contains test functions for the functions in the dataset_loader module.
It test the data preprocessing functions and steps.
It uses the pytest framework for unit testing.

Fixtures:
    - temp_data: A fixture to create temporary data for testing.

Test Functions:
    - test_clean_text(): Test function for the clean_text() function.
    - test_label_encoder(): Test function for the label_encoder() function.
    - test_preprocess_and_encode(temp_data, tmp_path): Test function for the \
        preprocess_and_encode() function.
    - test_get_dataset(temp_data): Test function for the get_dataset() function.
"""


import pytest
import polars as pl
from dataset_loader import (
    get_dataset,
    clean_text,
    label_encoder,
    preprocess_and_encode,
)


@pytest.fixture
def temp_dir_func(tmp_path):
    # Create a temporary directory for test files
    tmp_dir = tmp_path / "test_data"
    tmp_dir.mkdir()

    # Create a test Parquet file
    file_path = tmp_dir / "test_input.parquet"
    data = {
        "Target": [
            0,
            1,
            0,
        ],
        "Log": [
            "This is a test.",
            "Another test.",
            "Yet another test.",
        ],
    }
    test_df = pl.DataFrame(data)
    test_df.write_parquet(file_path, compression="gzip")
    return file_path, tmp_dir, data


def test_clean_text():
    # Test with a simple string
    result = clean_text("This is a simple test.")
    assert result == "simple test", f"result should be 'simple test', got {result}"

    # Test with HTML tags
    result = clean_text("<p>This is a <b>test</b>.</p>")
    assert result == "test", f"result should be 'test', got {result}"

    # Test with special characters
    result = clean_text("Clean this! *&^%$#@")
    # Check if the result is not an empty string
    assert result != "", "result should not be a empty string"

    # Test with stopwords
    result = clean_text("This is a test sentence with some stop words.")
    assert (
        result == "test sentence stop words"
    ), f"result should be 'test sentence stop words', got {result}"

    # Test with an empty string
    result = clean_text("")
    assert result == "", f"result should be empty srting, got {result}"

    # Test with all stopwords
    result = clean_text("the and is")
    assert result == "", f"result should be empty srting, got {result}"

    # Test with mixed case
    result = clean_text("This iS a MixEd CaSe TesT.")
    assert (
        result == "mixed case test"
    ), f"result should be 'mixed case test', got {result}"


def test_label_encoder():
    # Test with 'normal' label
    result_normal = label_encoder("normal")
    assert result_normal == 0, f"result should be 0, got {result_normal}"

    # Test with 'abnormal' label
    result_abnormal = label_encoder("abnormal")
    assert result_abnormal == 1, f"result should be 1, got {result_abnormal}"

    # Test with invalid label
    with pytest.raises(ValueError, match="Unrecognized label: unknown"):
        label_encoder("unknown"), "Raise wrong error or error message unmatched"

    """# Test with a DataFrame column
    df = pl.DataFrame({"target_column": ["normal", "abnormal", "normal"]})
    result_df = df.with_columns(pl.col("target_column")).apply(
        label_encoder, return_dtype=pl.Int16
    )
    expected_result_df = pl.DataFrame({"target_column": [0, 1, 0]})
    assert result_df.frame_equal(
        expected_result_df
    ), "result dataframe isnt equal to the expected preprocessed dataframe"""


def test_preprocess_and_encode(temp_dir_func):
    file_path, tmp_dir, _ = temp_dir_func
    # Create a save path for the processed data
    save_path = tmp_dir / "test_output.parquet"

    # Test the preprocess_and_encode function
    preprocess_and_encode(file_path=str(file_path), save_path=str(save_path))

    # Check if the processed file exists
    assert save_path.is_file(), "file doesnt exist and such wasn't saved"

    # Read the processed data to verify the changes
    processed_data = pl.read_parquet(str(save_path))

    # Verify label encoding
    expected_target = pl.Series(values=[0, 1, 0], name="Target", dtype=pl.Int16)
    assert processed_data["Target"].series_equal(
        expected_target
    ), "result Target dataframe isnt equal to the expected preprocessed label dataframe"

    # Verify text cleaning
    expected_log = pl.Series(values=["test", "test", "test"], name="Log")
    assert processed_data["Log"].series_equal(
        expected_log
    ), "result Log dataframe isnt equal to the expected preprocessed Log dataframe"


# @pytest.mark.parametrize("batch_s")
def test_get_dataset(temp_dir_func, batch_s=2):
    file_path, _, data = temp_dir_func
    # Test the get_dataset function
    dataset = get_dataset(
        file_path=str(file_path), batch_size=batch_s, shuffle_size=100, shuffle=True
    )

    # Verify the dataset structure
    for features, labels in dataset.take(1):
        assert features.shape == (
            batch_s,
        ), f"expected shape features of the TFDS to be '({batch_s},)', got {features.shape}"
        assert labels.shape == (
            batch_s,
        ), f"expected shape labels of the TFDS to be '({batch_s},)', got {labels.shape}"

    # Check if the dataset contains the expected number of elements
    # ceil(len(data) / batch_size)
    assert (
        len(list(dataset)) == (len(data["Target"]) + 1) // batch_s
    ), "the dataset contains the expected number of elements"
