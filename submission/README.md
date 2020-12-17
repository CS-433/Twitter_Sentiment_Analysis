# Sentiment Analysis of tweets: Code guide

## A. Overview

The python script `run.py` produces our best submission (#110091 on AIcrowd), see section (D).

Single models, as well as code needed by `run.py` to compute predictions, are contained in directories:

- `roberta/` containing all the code which trains and calculates predictions for RoBERTa-based pipelines
- `tfidf/` containing all the code which trains and calculates predictions for TF-IDF-based pipelines
- `glove/` containing all the code which trains and calculates predictions for GloVe-based pipelines
- `fast_text/` containing all the code which trains and calculates predictions for FastText-based pipelines

The directory `data/` is populated with our pretrained models, when they are downloaded.

Each of the above folders, contain the following files (assuming the model name is `[MODEL]`):

- `[MODEL]_predictions.ipynb` downloads our finetuned model and makes predictions for submitting to the challenge benchmark
- `[MODEL]_training.ipynb` performs the training of the model from scratch
- `[MODEL]_model.py` contains functions shared by the two Jupyter notebooks
- `[MODEL]_intermediate.py` produces the intermediate logits used by our stage2 classifier


## B. Dependencies

All the required libraries are listed in [TODO: Name of the file] and can be installed with the command `[TODO: COMMAND]`.

## C. Models info

We describe our pipelines in section III of our report.

Below, we detail the hardware requirements of different parts of our code:

| Model | GPU (predictions) | GPU (training) |
|:-----|:-----:|:-------:|
| TF-IDF | Not required | Not required |
| GloVe | _optional_ |**required**|
| FastText | Not required | Not required |
| RoBERTa | _optional_\* | **required** |

\* Obtaining test predictions without a GPU can take up to 30/40 mins

## D. `run.py`

We use the logits calculated by different _stage1_ models (TF-IDF + SVC, GloVe + LSTM, FastText, RoBERTa + Linear) to build the features vector fed to our _stage2_ classifier, which learns an optimal voting strategy.

When executing the script `run.py`, the code downloads our precomputed logits (one file for each _stage1_ model) and inputs them to the _stage2_ classifier, which computes labels. For reproducibility purposes, however, the logits can also be easily recalculated using our finetuned models (see section E).

## E. `[MODEL]_itermediate.py`

Each _stage1_ models implements, in its `[MODEL]_itermediate.py` file, the following two functions:

 1. `get_intermediate()`, which returns the precomputed logits.

 2. `generate_intermediate()`, which dowloads the finetuned model and uses it to calculate logits from scratch.
