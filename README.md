<a href="[https://opensource.org/licenses/MIT](https://github.com/psf/black)">
      <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
   </a>

# MultiNERD Named Entity Recognition (NER) Project

This repository contains code for training and evaluating a Named Entity Recognition (NER) model on the [MultiNERD dataset](https://huggingface.co/datasets/Babelscape/multinerd?row=17).

The goal is to develop two models that can identify and classify entities on
1. All 10 languages () and only 5 categories (Person, Organization, Location, Diseases and Anmial) 
2. All 15 categories, but only using the English language


## How to use
### Example Usage:
```bash
python main.py --learning-rate 1e-4 --batch-size 64 --epochs 5 --model-name bert-base-uncased --language-filter en
```
### Example Usage on 4 GPUs (1 node) using torchrun:
```bash
torchrun --nproc_per_node 4 main.py --gpu --learning-rate 5e-5 --model-name roberta-large --categories PER ORG
```

### Command-Line Arguments

- `--learning-rate`
   - *Description:* Learning rate for the optimizer.
   - *Type:* Float
   - *Default:* 5e-5
   - *Example:*
     ```bash
     python main.py --learning-rate 1e-4
     ```

- `--batch-size`
   - *Description:* Batch size for training.
   - *Type:* Integer
   - *Default:* 32
   - *Example:*
     ```bash
     python main.py --batch-size 64
     ```

- `--epochs`
   - *Description:* Number of training epochs.
   - *Type:* Integer
   - *Default:* 1
   - *Example:*
     ```bash
     python main.py --epochs 5
     ```

- `--model-name`
   - *Description:* Specify which model to fine-tune.
   - *Type:* String
   - *Default:* "prajjwal1/bert-tiny"
   - *Example:*
     ```bash
     python main.py --model-name "bert-base-uncased"
     ```

- `--language-filter`
   - *Description:* When specified, all other languages are filtered from the dataset.
   - *Type:* String
   - *Default:* None
   - *Example:*
     ```bash
     python main.py --language-filter en
     ```

- `--categories`
   - *Description:* When specified, all other categories are filtered from the dataset. Provide a whitespace-separated list of categories.
   - *Type:* List of Strings
   - *Default:* None
   - *Example:*
     ```bash
     python main.py --categories PERSON ORG LOC
     ```
NOTE: You can not use `--language-filter` together with `--categories`


## How to setup

### 1. Clone the Repository
```bash
git clone https://github.com/sandstromviktor/MultiNERD.git
cd multinerd-nerd
```
2.  Set Up Environment (Linux)
```bash
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

Or if you want to run a (Docker) container (May not work on GPU)
```bash
docker build -t multinerd -
docker run --rm -it -v $PWD/models:/home/code/models multinerd bash
```
This opens a shell to the container where you can run the same commands (see below) as you would in your venv.
The `-v` flag mounts the models folder to the repo folder so that your trained models are persistent on your drive.
## Training
The training script preprocesses the data and then uses the 🤗 `Trainer` to fine-tune the model.

### Fine-tune System A
System A is a language model that is trained to classify all entity types, but for only the english subset of the data. 

Run the following command to fine-tune a pre-trained language model on the English subset:
```bash
python main.py --model-name prajjwal1/bert-tiny --language-filter English
```
Specify model of your choice and set parameters as desired.

### Fine-tune System B
System B is a language model that is trained to classify five entity types, using all languages in the dataset. The assignment specifies the categories `PER ORG LOC DIS ANIM`

Run the following command to fine-tune a pre-trained language model on these categories. 
   ```bash
python main.py --model-name prajjwal1/bert-tiny --categories PER ORG LOC DIS ANIM 
```

## Evaluation

Evaluation of the model is done automatically in the training script every 1000 steps on the validation set. After training is completed, the model is evaulated on the test set.

### Metrics
This calculates Compute micro-F1, recall, precision and accuracy.
Let $M$ be the confusion matrix, then we define
$$F_{1, micro} = \frac{\sum_{i=1}^{n} M_{i,i}}{\sum_{i=1}^{n} M_{i,i} + \frac{1}{2}\left[\sum_{i=1}^{n} \sum_{j=1, j\neq j}^{n}  M_{i,j} + \sum_{i=1}^{n} \sum_{j=1, j\neq j}^{n}  M_{j,i} \right]}$$
This expression looks complicated (and rendered incorrectly on Github), but is basically the sum of all true positives (TP) divided by the sum of all TP plus half of the sum of all false positives (FP) plus the sum of all false negatives. 

## Experiments
Two experiments were conducted, each using the `bert-based-multilingual-cased` model ([Link](https://huggingface.co/bert-base-multilingual-cased)). Each model were trained for 1 epoch, using all default hyperparameters (see `train.py` for exact values) before tested on the `test-dataset`. 

Models were trained on 1 NVIDIA A-100 GPU (At NSC Berzelius)

### System A

```bash
python3 main.py --filter-language english --model-name bert-based-multilingual-cased
```

| Category            | F1                   | Precision            | Recall               | Number   |
|------------------|-------------------|-------------------|-------------------|----------|
| ANIM           | 0.598   | 0.816   | 0.472   | 32390   |
| BIO            | 0.333   | 0.592   | 0.232   | 250   |
| CEL            | 0.849   | 0.801   | 0.902   | 33900   |
| DIS            | 0.563   | 0.807   | 0.433   | 30676   |
| EVE            | 0.463   | 0.695   | 0.347   | 1406   |
| FOOD           | 0.949   | 0.948   | 0.950   | 6373068   |
| INST           | 0.889   | 0.865   | 0.914   | 145830   |
| LOC            | 0.471   | 0.503   | 0.444   | 28342   |
| MEDIA          | 0.590   | 0.592   | 0.589   | 11838   |
| MYTH           | 0.381   | 0.413   | 0.353   | 11032   |
| ORG            | 0.493   | 0.470   | 0.519   | 5994   |
| PER            | 0.927   | 0.909   | 0.945   | 169556   |
| PLANT          | 0.807   | 0.747   | 0.878   | 6484   |
| TIME           | 0.291   | 0.437   | 0.218   | 4106   |
| VEHI           | 0.693   | 0.754   | 0.641   | 11472   |
| All            | 0.940   | 0.940   |  0.939  | 6866344   |

## System B
```bash
python3 main.py --categories PER ORG LOC DIS ANIM --model-name bert-base-multilingual-cased
```

| Category            | F1                   | Precision            | Recall               | Number   |
|------------------|-------------------|-------------------|-------------------|----------|
| ANIM           | 0.801   | 0.811   | 0.792   | 28346     |
| DIS            | 0.964   | 0.972   | 0.956   | 138188      |
| LOC            | 0.983   | 0.982   | 0.985   | 169616     |
| ORG            | 0.953   | 0.948   | 0.957   | 33982     |
| PER            | 0.983   | 0.979   | 0.987   | 121334    |
| All            | 0.966   | 0.964   | 0.967   | 491466   |

## Summary

System A 

- Overall Performance: The model achieved an overall F1 score of 0.940 on the test dataset.
- Category-wise Analysis:
    - High Performers: The model excelled in categories such as FOOD (F1: 0.949), PER (F1: 0.927), and INST (F1: 0.889).
    - Challenges: Some categories like BIO (F1: 0.333) and TIME (F1: 0.291) posed challenges for the model.
    - Language Filter Impact: The language filtering with English subset did not negatively impact the model's performance, showcasing its adaptability across languages. This is probably due to the BERT model that has been previously trained on multiple languages. 

System B 

- Overall Performance: This model achieved a strong overall F1 score of 0.966 on the test dataset, surpassing System A in overall performance.
- Category-wise Analysis:
    - Consistency: The model demonstrated consistency across categories, with high F1 scores for ANIM, DIS, LOC, ORG, and PER.
    - High Performers: Notable performance in categories like LOC (F1: 0.983) and PER (F1: 0.983).
    - Multilingual Advantage: Training on a multilingual dataset proved beneficial, but could be due to the larger dataset.

General Comments:
- Data Size Impact: System B, trained on a larger multilingual dataset, outperformed System A in most categories.
- Category-specific Challenges: Some categories consistently posed challenges in both systems, indicating potential areas for further improvement, such as BIO and TIME.
- Label Normalization: There are indications of label normalization or spreading that may benefit from further investigation and refinement.
- Error Analysis: In-depth error analysis on misclassified instances can provide insights into specific challenges and guide targeted improvements.
