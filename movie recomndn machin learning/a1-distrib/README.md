
# Sentiment Classification 

assignment involves implementing and training sentiment classifiers on a labeled dataset of text sentences. The models implemented include a Perceptron Classifier and a Logistic Regression Classifier, with options to use Unigram, Bigram, and a combination of both as features. The objective is to classify text data into positive or negative sentiment categories.

## Repository Structure

- **data/**
  - `train.txt`: Training data containing sentences labeled with sentiment (positive/negative).
  - `dev.txt`: Development data used for model evaluation.
  - `test-blind.txt`: Test data for which predictions are generated (if required).
- **models.py**: Contains the implementation of various sentiment classifiers and feature extractors.
- **main.py**: The main script to run training and evaluation of the models.
- **sentiment_data.py**: Handles data loading and processing.
- **utils.py**: Contains utility functions, including an Indexer class used for feature indexing.

## Requirements

- Python 3.12.4
- NumPy
= NLTK
- SpaCy
- Other standard Python libraries


## How to Run the Code

### 1. Training and Evaluating a Model

To train and evaluate a model, use the `main.py` script. The script requires specifying the training data, development data, model type, and feature type. Below are examples of how to run the script:


#### Example 1: Logistic Regression with Unigram, BIGRAM & BETTER Features
CMD Promt or bash

python main.py --train_file data/train.txt --dev_file data/dev.txt --model LR --feats UNIGRAM
python main.py --train_file data/train.txt --dev_file data/dev.txt --model LR --feats BIGRAM
python main.py --train_file data/train.txt --dev_file data/dev.txt --model LR --feats BETTER


#### Example 2: Perceptron with Unigram, BIGRAM & BETTER Features

hon main.py --train_file data/train.txt --dev_file data/dev.txt --model PERCEPTRON --feats UNIGRAM
python main.py --train_file data/train.txt --dev_file data/dev.txt --model PERCEPTRON --feats BIGRAM
python main.py --train_file data/train.txt --dev_file data/dev.txt --model PERCEPTRON --feats BETTER



### 3. Output

Arguments: Namespace(train_file='data/train.txt', dev_file='data/dev.txt', model='LR', feats='UNIGRAM')
Reading training data from: data/train.txt
Number of training examples: 6920
Reading development data from: data/dev.txt
Number of development examples: 872
Training model with type: LR and features: UNIGRAM
Evaluating model...
Development set accuracy: 65.25%


## How to Approach the Assignment

### Step 1: I Understand the Problem

The goal is to classify sentences into positive or negative sentiment categories. The task involves implementing machine learning models and experimenting with different feature types.

### Step 2: Implement Models

- Implement a basic Perceptron and Logistic Regression model.
- Add support for Unigram and Bigram features. Optionally, combine them for better performance.

### Step 3: Train and Evaluate

- Use `train.txt` for training and `dev.txt` as provide
- Evaluate different configurations (models and feature types) to see which performs best.
