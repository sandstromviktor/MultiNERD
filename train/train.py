from datasets import load_dataset
from transformers import TrainingArguments

from span_marker import SpanMarkerModel, Trainer

categories = ["PER", "ORG", "LOC", "DIS", "ANIM"]


model_name = "bert-base-multilingual-cased"
model = SpanMarkerModel.from_pretrained(
    model_name,
    labels=labels,
    # SpanMarker hyperparameters:
    model_max_length=256,
    marker_max_length=128,
    entity_max_length=8,
)
    
# Prepare the 🤗 transformers training arguments
args = TrainingArguments(
    output_dir="models/span_marker_mbert_base_multinerd",
    # Training Hyperparameters:
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    # gradient_accumulation_steps=2,
    num_train_epochs=1,
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
    dataloader_num_workers=2,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)
trainer.train()