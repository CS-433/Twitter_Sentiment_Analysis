import wget
import pandas as pd
import pickle
root = 'data/'

def generate_intermediate(intermediate_filename= "tfidf_intermediate.csv"):
    
    # Load the embeddings
    print('Loading embeddings')
    embeddings_url = 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBclREZ3U5ejdJT1ZqcU16azJMUnMwRWhBQzIzakE_ZT1ub3Uy/root/content'
    embeddings_filename = 'data/tf-idf_test_embeddings.pkl'
    wget.download(embeddings_url, embeddings_filename)
    with open(embeddings_filename, 'rb') as file:
        X_test = pickle.load(file)

    # Load the trained classifier
    print('Loading classifier')
    # Load the trained classifier
    clf_url = 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBclREZ3U5ejdJT1ZqcU14dDhyUThpdWM1eDlFYXc_ZT1OZzFL/root/content'
    clf_filename = root + 'tf-idf_trained_linearSVC.pkl'
    wget.download(clf_url, clf_filename)
    with open(clf_filename, 'rb') as file:
        clf = pickle.load(file)
        
    # Make and save predictions
    print('Making predictions')
    df = pd.DataFrame(clf.decision_function(X_test), columns = ['Decision_function'])
    df.to_csv(intermediate_filename)
    print('Done')
    
def get_intermediate(intermediate_filename):
    """ Download pre-generated intermediate results to gain time. 
    
    Parameters
    -----------
        intermediate_filename: str, optional
            The path and name of the file with the intermediate data
    """
    
    url = 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBclREZ3U5ejdJT1ZqcU5BRWZtdnJEX1RSSm5kT0E_ZT1YVGIx/root/content'
    wget.download(url, intermediate_filename)
