import pytest
from data_utils.preprocess_data import preprocess_ner_tags
from datasets import load_dataset
import os

# Load your dataset for testing
cache_dir = os.path.join(os.getcwd(), "cache")
dataset = load_dataset("Babelscape/multinerd", cache_dir=cache_dir)


@pytest.mark.parametrize("input_row, expected_output", [
    ({"ner_tags": [1, 0, 3, 6, 0]}, {"ner_tags": [1, 0, 3, 6, 0]}),
    ({"ner_tags": [0, 0, 0, 0, 0, 0, 0, 0, 0]}, {"ner_tags": [0, 0, 0, 0, 0, 0, 0, 0, 0]}),
    ({"ner_tags": [13, 14, 15, 16, 17, 18, 1, 2, 3]}, {"ner_tags": [13, 14, 0, 0, 0, 0, 1, 2, 3]}),
    ({"ner_tags": [0, 0, 0, 0, 0, 0, 0, 0, 0]}, {"ner_tags": [0, 0, 0, 0, 0, 0, 0, 0, 0]}),
    ])
def test_preprocess_ner_tags(input_row, expected_output):
    # Perform the preprocessing
    processed_row = preprocess_ner_tags(input_row)

    # Verify that the ner_tags are processed correctly
    assert processed_row == expected_output


@pytest.mark.parametrize("input_row, expected_output", [
 ({"ner_tags": [1, "A", 3, None, "B"]}, ValueError),
    ({"ner_tags": ["X", "Y", "Z", 1, 2, 3]}, ValueError),
])
def test_preprocess_ner_tags_mixed_types(input_row, expected_output):
    with pytest.raises(expected_output):
        preprocess_ner_tags(input_row)


@pytest.mark.parametrize("input_row, expected_output", [
    ({"ner_tags": [5]}, {"ner_tags": [5]}),
])
def test_preprocess_ner_tags_single_element(input_row, expected_output):
    processed_row = preprocess_ner_tags(input_row)
    assert processed_row == expected_output