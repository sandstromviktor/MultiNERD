<a href="[https://opensource.org/licenses/MIT](https://github.com/psf/black)">
      <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
   </a>

# MultiNERD Named Entity Recognition (NER) Project

This repository contains code for training and evaluating a Named Entity Recognition (NER) model on the [MultiNERD dataset](https://huggingface.co/datasets/Babelscape/multinerd?row=17).

The goal is to develop two models that can identify and classify entities on
1. All 10 languages () and only 5 categories (Person, Organization, Location, Diseases and Anmial) 
2. All 15 categories, but only using the English language


## How to use

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
### Example Usage:
```bash
python main.py --learning-rate 1e-4 --batch-size 64 --epochs 5 --model-name "bert-base-uncased" --language-filter en
```

## How to setup

### 1. Clone the Repository
```bash
git clone https://github.com/sandstromviktor/MultiNERD.git
cd multinerd-nerd
```
2. Set Up Environment
```bash
python3 -m venv venv
pip install -r requirements.txt
```

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
This expression looks complicated, but is basically the sum of all true positives (TP) divided by the sum of all TP plus half of the sum of all false positives (FP) plus the sum of all false negatives. 

## Experiments
Two experiments were conducted, each using the `bert-based-multilingual-cased` model ([Link](https://huggingface.co/bert-base-multilingual-cased)). Each model were trained for 1 epoch, using all default hyperparameters (see `train.py` for exact values) before tested on the `test-dataset`. 

Models were trained on 1 NVIDIA A-100 GPU (At NSC Berzelius)

### System A

```bash
python3 main.py --filter-language english --model-name bert-based-multilingual-cased
```

| Category            | F1                   | Precision            | Recall               | Number   |
|------------------|-------------------|-------------------|-------------------|----------|
| ANIM           | 0.753   | 0.699   | 0.815   | 1852     |
| BIO            | 0.667   | 0.6     | 0.75    | 16       |
| CEL            | 0.982   | 0.983   | 0.980   | 6618     |
| DIS            | 0.969   | 0.969   | 0.969   | 916      |
| EVE            | 0.870   | 0.909   | 0.833   | 24       |
| FOOD           | 0.993   | 0.994   | 0.992   | 602916   |
| INST           | 0.998   | 0.998   | 0.998   | 11460    |
| LOC            | 0.764   | 0.733   | 0.799   | 3208     |
| MEDIA          | 0.771   | 0.773   | 0.771   | 1518     |
| MYTH           | 0.634   | 0.686   | 0.589   | 1144     |
| ORG            | 0.738   | 0.746   | 0.731   | 1004     |
| PER            | 0.995   | 0.994   | 0.996   | 24046    |
| PLANT          | 0.982   | 0.975   | 0.989   | 544      |
| TIME           | 0.596   | 0.627   | 0.568   | 366      |
| VEHI           | 0.966   | 0.958   | 0.974   | 704      |
| All            | 0.989   | 0.990   | 0.989   | 656336   |

## System B
```bash
python3 main.py --categories PER ORG LOC DIS ANIM --model-name bert-base-multilingual-cased
```