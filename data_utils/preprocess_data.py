import os
import logging
from datasets import load_dataset
from datasets import DatasetDict


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

LANGUAGE_MAP = {
    "chinese": "zh",
    "dutch": "nl",
    "english": "en",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "polish": "pl",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
}


def get_raw_dataset():
    cache_dir = os.path.join(os.getcwd(), "cache")
    logging.info(f"Downloading dataset and storing cache in {cache_dir}")
    return load_dataset("Babelscape/multinerd", cache_dir=cache_dir)


def get_single_languange_dataset(lang: str) -> DatasetDict:
    dataset = get_raw_dataset()
    logging.info(f"Fetching language filtered dataset")
    # Enable user to use either code or full language name (i.e., en, En, EnGlIsh, english etc)
    lang = lang.lower()
    lang = LANGUAGE_MAP.get(lang, lang)
    if lang not in LANGUAGE_MAP.values():
        raise ValueError(
            f"Invalid language. Expected one of {LANGUAGE_MAP.values()}, but got '{lang}'."
        )
    logging.info(f"Filtering on language: {lang}")
    dataset = dataset.filter(lambda set: set["lang"] == lang)
    logging.info("Filtering COMPLETE")
    return dataset


def get_category_filtered_dataset(categories: list) -> DatasetDict:
    dataset = get_raw_dataset()
    logging.info(f"Fetching category filtered dataset using categories: {categories}")
    tags = get_filtered_tags(categories)

    mapped_tags = set([TAG_MAP[tag] for tag in tags])

    logging.info(f"TAGS {mapped_tags}")
    return dataset.map(
        preprocess_ner_tags,
        batched=True,
        num_proc=4,
        fn_kwargs={"mapped_tags": mapped_tags},
    )


def preprocess_ner_tags(row: dict, mapped_tags: set) -> dict:
    ner_tags = row["ner_tags"]

    # Check if ner_tags is a list of lists
    if all(isinstance(inner_list, list) for inner_list in ner_tags):
        try:
            # Convert all inner values to integers
            ner_tags = [[int(value) for value in inner_list] for inner_list in ner_tags]
        except (ValueError, TypeError):
            raise ValueError(
                "Invalid value found in ner_tags. All values must be convertible to int."
            )

        # Replace values not in mapped_tags with 0 for each inner list
        row["ner_tags"] = [
            [0 if value not in mapped_tags else value for value in inner_list]
            for inner_list in ner_tags
        ]
    else:
        try:
            # Convert all values to integers
            ner_tags = [int(value) for value in ner_tags]
        except (ValueError, TypeError):
            raise ValueError(
                "Invalid value found in ner_tags. All values must be convertible to int."
            )

        # Replace values not in mapped_tags with 0 for the outer list
        row["ner_tags"] = [
            0 if value not in mapped_tags else value for value in ner_tags
        ]

    return row


def get_filtered_tags(categories: list) -> set:
    return set(
        [
            key
            for key, value in TAG_MAP.items()
            if any(substring in key for substring in categories)
        ]
    )


def get_all_tags() -> set:
    return set(TAG_MAP.keys())
