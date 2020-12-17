import wget
import pandas as pd
import pickle


def generate_intermediate(intermediate_filename= "tf-idf_intermediate.csv"):
    
    # Load the embeddings
    print('Loading embeddings')
    embeddings_filename = 'data/tf-idf_test_embeddings.pkl'
    with open(embeddings_filename, 'rb') as file:
        X_test = pickle.load(file)

    # Load the trained classifier
    print('Loading classifier')
    clf_filename = 'data/tf-idf_trained_linearSVC.pkl'
    with open(clf_filename, 'rb') as file:
        clf = pickle.load(file)
        
    # Make and save predictions
    print('Making predictions')
    df = pd.DataFrame(clf.decision_function(X_test), columns = ['Decision_function'])
    df.to_csv(intermediate_filename)
    print('Done')