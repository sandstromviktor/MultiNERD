from typing import Type, Any
import logging
from argparse import ArgumentParser
from datasets import load_dataset
from transformers import TrainingArguments

from span_marker import SpanMarkerModel, Trainer

from data_utils.preprocess_data import get_filtered_tags, get_all_tags, get_single_languange_dataset, get_category_filtered_dataset


POSSIBLE_MODELS = ["prajjwal1/bert-tiny", 
          "prajjwal1/bert-mini", 
          "prajjwal1/bert-small", 
          "prajjwal1/bert-medium",
          'bert-base-multilingual-cased', 
          'roberta-large']


def get_model(model_name: str, categories: list = None) -> SpanMarkerModel:
    if model_name not in POSSIBLE_MODELS:
        raise ValueError(f"Invalid model. Expected one of {POSSIBLE_MODELS}, but got '{model_name}'.")
    
    if categories:
        labels = get_filtered_tags(categories).union({'O'})
    else:
        labels = get_all_tags()
    
    model = SpanMarkerModel.from_pretrained(
        model_name,
        labels=labels,
        # SpanMarker hyperparameters:
        model_max_length=256,
        marker_max_length=128,
        entity_max_length=10,
    )
    logging.info(f"Pretrained model is set to {model_name}")
    return model

def get_training_args(args: ArgumentParser) -> TrainingArguments:

    # Prepare the 🤗 transformers training arguments
    train_args = TrainingArguments(
        output_dir="models/span_marker_mbert_base_multinerd",
        # Training Hyperparameters:
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,

        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        bf16=True,  # Replace `bf16` with `fp16` if your hardware can't use bf16.
        # Other Training parameters
        logging_first_step=True,
        logging_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_total_limit=2,
        dataloader_num_workers=4,
    )
    return train_args



def train(args):
    
    model = get_model(args.model_name, args.categories)
    train_args = get_training_args(args)
    
    if args.categories:
        dataset = get_category_filtered_dataset(args.categories)
    
    if args.language_filter:
        dataset = get_single_languange_dataset(args.language_filter)
    
    
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    trainer.train()