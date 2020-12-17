######################################################
# WARNING!                                           #
# GPU is required to run 'generate_intermediate()'   #
######################################################
import transformers
from transformers import AutoTokenizer, RobertaForSequenceClassification
import numpy as np
import requests
import wget

# Contains preprocessing functions
from roberta.preprocessing_v6 import *
# Contains all the functions related to the model
from roberta.roberta_model import *

def download_resource(local_name, url):
  resp = requests.get(url, allow_redirects=True)
  open(local_name, "wb").write(resp.content)

MAX_LENGTH = 200   
BATCH_SIZE = 32

def generate_intermediate(intermediate_filename="roberta_intermediate.csv"):
  """
  Generate the intermediate data used by stage2 classifier
  
  Parameters
  -----------
  intermediate_filename: str, optional
    The path and name of the file with the intermediate data
  """
  if not torch.cuda.is_available():
    raise Error("CUDA-enabled device needed to calculate logits")

  used_device = torch.device('cuda:0')

  # Load tokenizer
  bert_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
  # Load model architecture
  bert_model = RobertaForSequenceClassification.from_pretrained("roberta-base")

  # Download the finetuned model
  download_resource("RoBERTa_finetuned_std.pth", "https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBclREZ3U5ejdJT1ZqcU1ndVZGUzV0RDdJeTFIUUE_ZT1wRW9q/root/content")
  # Load refined weights
  reloaded_model = load_model("RoBERTa_finetuned_std", bert_model, used_device)

  # Download test_data.txt
  download_resource("test_data.txt", "https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3QvcyFBclREZ3U5ejdJT1ZqcDR5Q3hoWXM4T2FJd1JLenc_ZT1hSXh0/root/content")
  
  ordered_input = []
  ordered_labels = []

  with open("test_data.txt", "r") as f:
      for line in f.readlines():
          comma_pos = line.find(",")
          ordered_labels.append(int(line[:comma_pos]))
          ordered_input.append(line[comma_pos+1:])

  ordered_input = np.array(ordered_input)
  ordered_labels = np.array(ordered_labels)

  # Sanity check
  assert ordered_input.shape[0] == 10_000

  ordered_dataset = SentimentDataset(
      chunks=ordered_input,
      labels=ordered_labels,
      tokenizer=bert_tokenizer,
      max_len=MAX_LENGTH
  )

  ordered_loader = get_loader(ordered_dataset, BATCH_SIZE)

  # Generate logits
  sub_idxs, sub_labels, sub_logits = prepare_submission(reloaded_model, bert_tokenizer, used_device, BATCH_SIZE, max_len=MAX_LENGTH, test_filename="test_data.txt")

  # Save intermediate result to file
  np.savetxt(intermediate_filename, np.concatenate([sub_idxs[...,np.newaxis], sub_logits], axis=1), fmt=['%d', '%f', '%f'], delimiter=',', header="Idx,Logit_zero,Logit_one", comments="")
  
  # Free GPU memory
  del reloaded_model
  torch.cuda.empty_cache()

def get_intermediate(intermediate_filename):
    """ Download pre-generated intermediate results to gain time. 
    
    Parameters
    -----------
        intermediate_filename: str, optional
            The path and name of the file with the intermediate data
    """
    
    url = 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBclREZ3U5ejdJT1ZqcU0tUGZ3MWhwZHhUVHkweGc_ZT1odVlq/root/content'
    wget.download(url, intermediate_filename)
