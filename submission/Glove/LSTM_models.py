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