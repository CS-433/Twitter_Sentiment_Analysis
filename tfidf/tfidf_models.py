import numpy as np 
import pandas as pd 
from nltk.tokenize import TweetTokenizer
import re
from wordsegment import load, segment
load()

### Helpers
def txt_to_list(filename):
    """ Extracts a text file into a list of tweets.
    
    Parameters
    ----------
    filename: string
        Relative path of text file
        
    Returns
    -------
    tweets: list of strings 
        a list of all tweets in the file 
    """
    
    tweets = []
    with open(filename, encoding = 'utf-8') as f:
        for line in f:
            tweets.append(line[:-1]) # Remove '\n'
            
    return tweets

def merge_shuffle_label(positive, negative, seed = 0):
    """ Merge positive and negative tweets, shuffle to create a dataset and provide appropriate labels. 
    
    Parameters
    ----------
    positive: list of strings
        list of positive tweets
    negative: list of strings
        list of negative tweets
    seed: int, optional
        seed to be passed to numpy.random before shuffling
        
    Returns 
    -------
    all_tweets: 1D numpy array of strings
        All tweets, shuffled
    y: 1D numpy array (0 or 1)
        corresponding labels of tweets
    """
    
    all_tweets = np.concatenate((positive, negative))
    y = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))

    np.random.seed(seed)
    random_idxs = np.random.permutation(len(y))

    all_tweets = all_tweets[random_idxs]
    y = y[random_idxs].astype(int)
    
    return all_tweets, y

def split_dataset(fraction, tweets, y):
    """ Split dataset into training and validation.  
    
    Note: tweets are expected to have been shuffled.
    
    Parameters
    ----------
    fraction: float in ]0, 1[
        fraction of training observations
    tweets: 1D numpy array of strings
        all training tweets
    y: 1D numpy array of integers (0 or 1)
        corresponding labels 
        
    Returns
    -------
    train: 1D numpy array of strings
        tweets to keep for training 
    y_train: 1D numpy array of integers (0 or 1)
        corresponding labels
    val: 1D numpy array of strings
        tweets to keep for validation 
    y_val: 1D numpy array of integers (0 or 1)
        corresponding labels 
    """
    
    N_train = int(fraction*len(y))

    train, val = tweets[:N_train], tweets[N_train:]
    y_train, y_val = y[:N_train], y[N_train:]
    
    return train, val, y_train, y_val 

def judge_pred(classifier, xtr, xv, ytr, yv):
    """ Measure performance on training and validation sets. 
    
    Parameters
    ----------
    classifier: sklearn classifier
        classifier to judge
    xtr: numpy array or sparse matrix of shape (N, D)
        training set
    xtr: numpy array or sparse matrix of shape (N, D)
        validation set
        relative path of csv file
    ytr: 1D numpy array of (0, 1)
        true labels for training set
    yv: 1D numpy array of (0, 1)
        true labels for validation set
    """
    
    train_acc = (classifier.predict(xtr) == ytr).mean()
    val_acc = (classifier.predict(xv) == yv).mean()
    print('Training set accuracy: {:.2f}% / validation set: {:.2f}%'.format(100*train_acc, 100*val_acc))
    
def save_pred(filename, predictions):
    """ Save (0, 1) predictions in the desired csv format for AIcrowd, with (-1, 1) labels. 
    
    Parameters
    ----------
    filename: string
        relative path of csv file
    predictions: 1D numpy array of (0, 1)
        predictions for unseen test set
    """
    
    preds = pd.DataFrame((2*predictions-1).astype(int), columns = ['Prediction'], index = np.arange(1, len(predictions)+1))
    preds.index.names = ['Id']
    preds.to_csv(filename)


### Pre-processing

def split_hashtag(before):
    """ Split a token if it is a hashtag (separating words after '#') """
    if len(before) == 0:
        return ""
    
    if before[0] == "#":
        return ' '.join(segment(before))
    return before

def remove_repeats(text):
    """ Replace repeated letters by single letter. """
    return re.sub(r'([a-z])\1+', r'\1', text)

def to_vec(lmt_wise_method):
    """ Make element-wise operation application to iterables. """
    return np.vectorize(lmt_wise_method)

def process_sentence(sentence_array, purge_methods):
    """ Apply a sequence of pre-processing methods to a list of tokens. """
    for method in purge_methods:
        sentence_array = method(sentence_array)
    return sentence_array

preproc_pipeline = [to_vec(split_hashtag),  
                    to_vec(remove_repeats)]

def tk(sent):
    """ Tokenize a tweet.
    
    Parameters
    ----------
        sent: string
            a tweet
        
    Returns
    -------
        tokens: list of strings
            a tokenized version of the string
    """
    tokens = TweetTokenizer().tokenize(sent)
    tokens = process_sentence(tokens, preproc_pipeline)
    return tokens