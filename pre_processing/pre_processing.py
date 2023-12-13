from datasets import load_dataset
import os
from typing import Type, Any

cache_dir = os.path.join(os.getcwd(), "cache")
dataset = load_dataset("Babelscape/multinerd", cache_dir=cache_dir)

TAG_MAP = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-ANIM": 7,
    "I-ANIM": 8,
    "B-BIO": 9,
    "I-BIO": 10,
    "B-CEL": 11,
    "I-CEL": 12,
    "B-DIS": 13,
    "I-DIS": 14,
    "B-EVE": 15,
    "I-EVE": 16,
    "B-FOOD": 17,
    "I-FOOD": 18,
    "B-INST": 19,
    "I-INST": 20,
    "B-MEDIA": 21,
    "I-MEDIA": 22,
    "B-MYTH": 23,
    "I-MYTH": 24,
    "B-PLANT": 25,
    "I-PLANT": 26,
    "B-TIME": 27,
    "I-TIME": 28,
    "B-VEHI": 29,
    "I-VEHI": 30,
  }


def get_single_languange_dataset(language: str) -> Type[Any]:
    return dataset["validation"].filter(lambda set: set["lang"] == language)


def get_category_filtered_dataset(categories: list) -> Type[Any]:
    tags = get_filtered_tags(categories)
    mapped_tags = set([TAG_MAP[tag] for tag in tags])
    return dataset.map(preprocess_ner_tags, mapped_tags)


def preprocess_ner_tags(row: dict, mapped_tags: set) -> dict:
    ner_tags = row["ner_tags"]
    try:
        ner_tags = [int(value) for value in ner_tags]
    except (ValueError, TypeError):
        raise ValueError("Invalid value found in ner_tags. All values must be convertible to int.")
    
    row["ner_tags"] = [0 if value not in mapped_tags else value for value in ner_tags]
    return row


def get_filtered_tags(categories: list) -> set:
    return set([key for key, value in TAG_MAP.items() if any(substring in key for substring in categories)])
