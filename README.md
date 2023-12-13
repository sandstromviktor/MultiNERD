# MultiNERD Named Entity Recognition (NER) Project

This repository contains code for training and evaluating a Named Entity Recognition (NER) model on the MultiNERD dataset. The goal is to develop a model that can identify and classify entities in English text into five categories: PERSON, ORGANIZATION, LOCATION, DISEASES, ANIMAL, and the O tag (not part of an entity).

## Steps to Reproduce

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/multinerd-nerd.git
cd multinerd-nerd
```
2. Set Up Environment
```bash
pip install -r requirements.txt
```
3. Download and Preprocess Data

Download the MultiNERD dataset and preprocess it to filter out non-English examples.
4. Fine-tune System A

Fine-tune a pre-trained language model (LM) on the English subset of the training set.
5. Preprocess Data for System B

Perform necessary pre-processing to predict only five entity types (PERSON, ORGANIZATION, LOCATION, DISEASES, ANIMAL) and the O tag.
6. Fine-tune System B

Fine-tune the model on the filtered dataset created in step 5.
7. Evaluate Models

Choose a suitable metric and evaluate both System A and System B using the test set.
Results and Findings

In our experiments, we observed [insert main findings here]. However, it's essential to note that [mention any limitations or challenges encountered during the experiments].
Repository Structure

    src/: Contains the source code for data preprocessing, model training, and evaluation.
    data/: Placeholder for storing the dataset (not included in the repository).
    models/: Placeholder for saving the trained models.
    requirements.txt: Lists the dependencies needed to run the code.