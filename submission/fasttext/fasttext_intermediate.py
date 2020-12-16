import wget
import pandas as pd
import pickle
import fasttext


def generate_intermediate(intermediate_filename= "tf-idf_intermediate.csv"):
    """ Generate the intermediate data used by stage2 classifier
    Parameters
    -----------
        intermediate_filename: str, optional
            The path and name of the file with the intermediate data
    """
    
    # Load the model
    root = 'data/'
    model = fasttext.load_model(root + "fasttext_trained_model.bin")
    
    # Prepare test set
    test_url = 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3QvcyFBclREZ3U5ejdJT1ZqcDR5Q3hoWXM4T2FJd1JLenc_ZT1hSXh0/root/content'
    test_filename = root + 'test.txt'
    wget.download(test_url, test_filename)

    test_tweets = []
    with open(test_filename, encoding = 'utf-8') as f:
        for line in f:
            sp = line.split(',')

            test_tweets.append(','.join(sp[1:])[:-1]) # Remove index and \n

    # Generate predictions
    pred = model.predict(test_tweets, k=1)
    confidence = [el[0] for el in pred[1]]
    res = {'__label__0': 0, '__label__1': 1}
    predicted_label = [res[el[0]] for el in pred[0]]

    df = pd.DataFrame(list(zip(confidence, predicted_label)), 
                   columns =['confidence', 'predicted_label'])

    # Save predictions
    save_filename = 'fasttext_test_confidence.csv'
    df.to_csv(save_filename)

   
    