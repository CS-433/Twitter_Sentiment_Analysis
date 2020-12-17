import pandas as pd
import numpy as np


def load_all_intermediate(fts_path, tfidf_path, roberta_path = None, glove_path = None):
	""" Koad intermediate results from other models for final predictions. 
	
	Parameters
	----------
		fts_path: string
			path to fastText intermediate predictions
		tfidf_path: string
			path to fastText intermediate predictions
		roberta_path: string
			path to fastText intermediate predictions
		glove_path: string
			path to fastText intermediate predictions
	
	Returns
	-------
		scores_t: pd.DataFrame
			Dataframe with merged intermediate predictions
	"""
	
	scores_t = pd.read_csv(fts_path).drop(columns = ['Unnamed: 0']).rename(columns = {'confidence': 'fts_confidence', 'predicted_label': 'fts_label'})
	
	tfidf_t = pd.read_csv(tfidf_path).drop(columns = ['Unnamed: 0']).rename(columns = {'Decision_function': 'tfidf_decision'})
	scores_t = pd.merge(scores_t, tfidf_t, left_index = True, right_index = True)
	
	roberta_t = pd.read_csv(roberta_path).drop(columns = 'Idx').rename(columns = {'Real': 'y', 'Logit_zero': 'roberta_logit0','Logit_one': 'roberta_logit1' })
	scores_t = pd.merge(scores_t, roberta_t, left_index = True, right_index = True)
	
	glove_t = pd.read_csv(glove_path).drop(columns = 'Unnamed: 0').rename(columns = {'prediction': 'glove_p'})
	scores_t = pd.merge(scores_t, glove_t, left_index = True, right_index = True)
	
	return scores_t

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
