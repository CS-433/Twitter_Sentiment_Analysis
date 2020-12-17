import wget
root = 'data/'


#Since we wasn't able to save the LSTM model, submission file is downloaded directly 
def generate_intermediate(intermediate_filename= "glove_intermediate.csv"):
     
    
    submission_url = 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBclREZ3U5ejdJT1ZqcU02NlRkVzNJMnVvbkRRUHc_ZT00OUVm/root/content'
    
    wget.download(submission_url, intermediate_filename)
