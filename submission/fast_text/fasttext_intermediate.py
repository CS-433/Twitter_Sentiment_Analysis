import wget
import pandas as pd
import pickle
import fasttext as ft


def generate_intermediate(intermediate_filename= "fasttext_intermediate.csv"):
    """ Generate the intermediate data used by stage2 classifier
    Parameters
    -----------
        intermediate_filename: str, optional
            The path and name of the file with the intermediate data
    """
    
    # Load the model
    root = 'data/'
    ## OLD model_url = 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBclREZ3U5ejdJT1ZqcU0yOGxGTDgya0l4OGNlNGc_ZT1DMnNy/root/content'
    model_url = 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBclREZ3U5ejdJT1ZqcU0zb1h1eUhDbkpfWHRqS0E_ZT0xblV3/root/content'
    model_filename = root + 'fasttext_trained_model.bin'
    wget.download(model_url, model_filename)

    model = ft.load_model(model_filename)
    
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
    df.to_csv(intermediate_filename)
