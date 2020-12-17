import wget
root = 'data/'

# The LSTM could not be saved. Please generate intermediate results directly in training. 
 
def get_intermediate(intermediate_filename):
    """ Download pre-generated intermediate results to gain time. 
    
    Parameters
    -----------
        intermediate_filename: str, optional
            The path and name of the file with the intermediate data
    """
    
    url = 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBclREZ3U5ejdJT1ZqcU1feVFjSHhEMEdaYzBpUGc_ZT1xOXpM/root/content'
    wget.download(url, intermediate_filename)
