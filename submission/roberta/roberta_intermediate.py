######################################################
# WARNING!                                           #
# GPU is required to run 'generate_intermediate()'   #
######################################################
import transformers
from transformers import AutoTokenizer, RobertaForSequenceClassification
import numpy as np
import urllib

# Contains preprocessing functions
from preprocessing_v6 import *
# Contains all the functions related to the model
from roberta_model import *

def dowload_resource(local_name, url):
   urllib.urlretrieve(url, filename=local_name)

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
  with open("test_data.txt", "r") as f:
      for line in f.readlines():
          ordered_input.append(line)

  ordered_input = np.array(ordered_input)

  # Sanity check
  assert ordered_input.shape[0] == 10_000

  ordered_dataset = SentimentDataset(
      chunks=ordered_input,
      labels=np.,
      tokenizer=bert_tokenizer,
      max_len=MAX_LENGTH
  )

  ordered_loader = get_loader(ordered_dataset, BATCH_SIZE)

  # Generate logits
  sub_idxs, sub_labels, sub_logits = prepare_submission(reloaded_model, bert_tokenizer, used_device, BATCH_SIZE, max_length=MAX_LENGTH, test_filename="test_data.txt")

  # Save intermediate result to file
  np.savetxt(intermediate_filename, np.concatenate([sub_idxs[...,np.newaxis], sub_logits], axis=1), fmt=['%d', '%f', '%f'], delimiter=',', header="Idx,Logit_zero,Logit_one", comments="")
  
  # Free GPU memory
  del reloaded_model
  torch.cuda.empty_cache()