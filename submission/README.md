# Sentiment Analysis of tweets: Code guide

## A. Overview

The python script `run.py` produces our best submission (#[!!!!!! TODO:PUT NUMBER] on AIcrowd), based on the stage2 classifier

Single models, as well as code needed by `run.py` to compute predictions, are contained in directories:

- `roberta/` containing all the code which trains and calculates predictions for RoBERTa-based pipelines
- `tfidf/` containing all the code which trains and calculates predictions for TF-IDF-based pipelines
- `glove/` containing all the code which trains and calculates predictions for GloVe-based pipelines
- `fasttext/` containing all the code which trains and calculates predictions for FastText-based pipelines

Each of the above folders, contain the following files (assuming the model name is `[MODEL]`):

- `[MODEL]_predictions.ipynb` downloads our finetuned model and makes predictions for submitting to the challenge benchmark
- `[MODEL]_training.ipynb` performs the training of the model from scratch
- `[MODEL]_model.py` contains functions shared by the two Jupyter notebooks
- `[MODEL]_intermediate.py` produces the intermediate logits used by our stage2 classifier


## B. Dependencies

TODO: Add list of libraries used
(Remember about urllib)

## TODO: ADD other details below!! (Still have old description of project1)

## B. `Model.py`

### B.1. Structure

The functions used by our ML model are divided in 5 categories:

1. __General helper functions__: Automating small but heavily used tasks (such as accuracy measurement or expansion of a value into a dictionary indexed by jet numbers)
2. __Data Splitting__: Performing cross-validation related tasks
3. __Data Cleaning__: Handling the removal of bad values from the data and their normalization
4. __Training__: Taking care of fitting the model weights to the training data
5. __Feature expansion__: Providing ways to generate new features starting from the ones in the data

### B.2. Data cleaning and expansion

Samples for every jet number class are treated separately:

1. The features which contain only -999 values or have 0 variance are discarted, since they provide no useful information (operation performed by `clean_data()`)
2. The __same__ set of transformations is applied to the remaining features in each `jet_num` class. They are:
  
    - polynomial expansion
    - calculation of the tangent
    - ranking
    - interaction terms

    This is done by `expand()`

### B.3. Cleaning __clues__

Many parameters should be calculated in order to clean and expand the data. While this is good for TRAIN data, those parameters should not be calculated from TEST data.
For that reason, during TRAIN data processing, we save __clues__ which are information used during TEST data processing.
Many functions such as `expand()` and `clean_data()` use some form of clues.
