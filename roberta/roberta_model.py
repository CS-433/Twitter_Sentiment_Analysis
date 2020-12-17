import torch.nn as nn
import torch
from roberta.preprocessing_v6 import *

def apply_preprocessing(bert_tokenizer, tweet):
    return " ".join(process_sentence(tweet.split(" "), standard_pipeline(bert_tokenizer)))

class RobertaSimple(nn.Module):
    """
    Simple encapsulator of a RoBERTa model
    """
    def __init__(
            self,
            bert_model
    ):
        super(RobertaSimple, self).__init__()
        self.model = bert_model

    def forward(self, input_ids, input_attention, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=input_attention, labels=labels)
        
        return outputs
    
class SentimentDataset(torch.utils.data.Dataset):
    """
    Dataset object providing properly encoded samples
    """
    def __init__(self, chunks, labels, tokenizer, max_len):
        self.chunks = chunks
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return self.chunks.shape[0]
    
    def __getitem__(self, item):
        sentence = self.chunks[item]
        labels = self.labels[item]
        
        encoded = self.tokenizer.encode_plus(
            apply_preprocessing(self.tokenizer, sentence),
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

def get_loader(dataset, batch_size):
    """
    Converts a Dataset into a DataLoader
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0
    )

def save_model(filename, model):
    """
    Save the model weights to file
    NOTE: All training-related information is lost. Therefore the training cannot
    be resumed, based on this file

    Parameters
    ----------
    filename: str, required
        The name of the file (without the extension)
    model : RobertaSimple, required
        The model
    """
    torch.save(model.state_dict(), filename + ".pth")
    print("Model saved")
    
def load_model(filename, bert_model, device):
    """
    Loads the RobertaSimple model stored in a pth file
    NOTE: Only the weights are stored. Partial training
    cannot be resumed after loading the model from file
    
    Params
    --------
    filename: str, required
        The name of the file which stores the weights (without extension)
    bert_model: required
        The core version of RoBERTa from 'transformers'
    device: required
        The processing unit in which to load the model
        
    Returns
    --------
    model:
        The loaded model
    """
    model = RobertaSimple(bert_model)
    model = model.to(device)
    model.load_state_dict(torch.load(filename + ".pth", map_location=device))
    model.eval()
    print("Model loaded")
    return model

def eval_model(model, data_loader, device):
    """
    Returns the predictions of 'model' on data contained in 'data_loader'

    Parameters
    ----------
    model : RobertaSimple, required
        The model
    data_loader: DataLoader, required
        The loader providing data
    device: required
        The device which executes the computations

    Returns
    -------
    accuracy: float
    mean_loss: float
    """
    model = model.eval()
    losses = []
    correct_predictions = 0
    num_preds = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                input_attention=attention_mask,
                labels=targets
            )

            loss = outputs.loss
            logits = outputs.logits
            
            preds = torch.argmax(logits, dim=1)
            
            correct_predictions += torch.sum(preds == targets)
            num_preds += targets.shape[0]
            losses.append(loss.item())
    return correct_predictions.double() / float(num_preds), np.mean(losses)

def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
    """
    Trains the model for 1 epoch

    Parameters
    ----------
    model : RobertaSimple, required
        The model
    data_loader: DataLoader, required
        The loader providing data
    optimizer: required
        The optimizer
    device: required
        The device which executes the computations
    scheduler: required
        The scheduler
    n_examples: required
        The size of the entire dataset

    Returns
    -------
    running_accuracy: float
    mean_loss: float
    """
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)

        outputs = model(
          input_ids=input_ids,
          input_attention=attention_mask,
          labels=targets
        )

        logits = outputs.logits
        loss = outputs.loss

        preds = torch.argmax(logits, dim=1)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)
    
def predict(model, data_loader, device):
    """
    Generates the predictions by the model on data
    
    Parameters
    ----------
    model: RobertaSimple, required
        The model
    data_loader: DataLoader, required
        The loader providing data
    device:
        The device which exectutes the computation

    Returns
    -------
    idxs: 1D Numpy array
        Contains the ids of the data
    preds: 1D Numpy array
        Contains the predictions (0 or 1 label, for each datapoint)
    logits: 2D Numpy array (N, 2)
        Contains the logits, 2 floats for every id
    """
    model = model.eval()

    idxs = []
    predictions = []
    logits_list = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)

            idx = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                input_attention=attention_mask,
                labels=torch.zeros(idx.shape[0], dtype=torch.long).to(device),
            )

            logits = outputs.logits
            loss = outputs.loss
            
            preds = torch.argmax(logits, dim=1)

            idxs.append(idx.cpu())
            predictions.append(preds.cpu())
            logits_list.append(logits.cpu())

    return np.concatenate(idxs), np.concatenate(predictions), np.concatenate(logits_list)

def prepare_submission(model, tokenizer, device, batch_size, max_len=200, test_filename="test_data.txt"):
    """
    Loads test data and generates predictions
    
    Parameters
    ----------
    model: RobertaSimple, required
        The model
    device: required
        The device which exectutes the computation
    max_len: int, optional
        The maximum number of tokens for each datapoint
    test_filename: str, optional
        The name of the file containing the test data

    Returns
    -------
    idxs: 1D Numpy array
        Contains the ids of the data
    preds: 1D Numpy array
        Contains the predictions (0 or 1 label, for each datapoint)
    logits: 2D Numpy array (N, 2)
        Contains the logits, 2 floats for every id
    """
    print("Loading file...")
    unk_ids = []
    unk_data = []
    with open(test_filename, "r") as f:
        for line in f.readlines():
            comma_pos = line.find(",")
            unk_ids.append(int(line[:comma_pos]))
            unk_data.append(line[comma_pos+1:])
            
    # Sanity check
    assert len(unk_data) == 10000

    print("Content:", unk_ids[:2], unk_data[:2])
    
    print("Create dataloader...")
    dataset = SentimentDataset(
        np.array(unk_data), 
        np.array(unk_ids), 
        tokenizer=tokenizer, 
        max_len=max_len
    )
    
    d_loader = get_loader(dataset, batch_size)

    print("Generating predictions...")
    return predict(model, d_loader, device)

def write_submission(filename, idxs, labels):
    """
    Saves the predictions to file, in the format required for submission
    
    Parameters
    ----------
    filename: str, required
        The name of the output file
    idxs: 1D array
        The list of ids
    labels: 1D array
        The list of predictions (1 for each id)
        
    Returns
    ----------
    None
    """
    labels = (labels * 2 - 1).astype(int)
    idxs = idxs.astype(int)
    submission_content = np.concatenate([idxs[..., np.newaxis], labels[..., np.newaxis]], axis=1).astype(int)
    print(submission_content)
    np.savetxt(filename, submission_content, fmt='%d', delimiter=',', header="Id,Prediction", comments="")
