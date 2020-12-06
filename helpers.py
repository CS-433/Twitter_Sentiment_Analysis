import numpy as np 
import pandas as pd 


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
            tweets.append(line)
            
    return tweets

def merge_shuffle_label(positive, negative):
    """ Merge positive and negative tweets, shuffle to create a dataset and provide appropriate labels. 
    
    Parameters
    ----------
    positive: list of strings
        list of positive tweets
    negative: list of strings
        list of negative tweets
        
    Returns 
    -------
    all_tweets: 1D numpy array of strings
        All tweets, shuffled
    y: 1D numpy array (0 or 1)
        corresponding labels of tweets
    """
    
    all_tweets = np.concatenate((positive, negative))
    y = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))

    random_idxs = np.random.permutation(len(y))

    all_tweets = all_tweets[random_idxs]
    y = y[random_idxs].astype(int)
    
    return all_tweets, y
    
    
def write_labeled(filename, tweets, y):
    """ Write a labeled dataset to text file.  
    
    Parameters
    ----------
    filename: string
        relative path to save
    tweets: 1D numpy array of strings
        tweets to save
    y: 1D numpy array of integers (0 or 1)
        corresponding labels 
    """
    
    
    with open(filename, 'w', encoding = 'utf-8') as f:
        for i, tweet in enumerate(tweets):
            f.write('__label__{} '.format(y[i]) + line)
        
        
def split_dataset(fraction, tweets, y):
    """ Split dataset into training and validation.   
    
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