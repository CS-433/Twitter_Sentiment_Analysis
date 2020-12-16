import wget
import pandas as pd
import pickle


def generate_intermediate(intermediate_filename= "tf-idf_intermediate.csv"):
    """ Generate the intermediate data used by stage2 classifier
    Parameters
    -----------
        intermediate_filename: str, optional
            The path and name of the file with the intermediate data
    """
    
    # Prepare test data
    """print('Loading test data') # Not needed if load embeddings
    test_url = 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3QvcyFBclREZ3U5ejdJT1ZqcDR5Q3hoWXM4T2FJd1JLenc_ZT1hSXh0/root/content'
    local_name = 'test_data.txt'
    wget.download(test_url, local_name)
    test_tweets = []
    with open(local_name, encoding = 'utf-8') as f:
        for line in f:
            sp = line.split(',')
            index = sp[0]
            test_tweets.append(','.join(sp[1:]))
    print('Loading vectorizer')
    main()
    vectorizer_filename = 'data/tf-idf_fitted_vectorizer.pkl'
    with open(vectorizer_filename, 'rb') as file:
        vect = pickle.load(file)  
    X_test = vect.transform(test_tweets)
     """ 
    
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
    

   
    