import numpy as np

# Adapted from here: https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions
contractions = { 
"ain't": "be not", # "am not / are not / is not / has not / have not",
"aren't": "be not", # "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would", # "he had / he would",
"he'd've": "he would have",
"he'll": "he will", # "he shall / he will",
"he's": "he is", # "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"howdy": "how do you",
"how'll": "how will",
"how's": "how", # "how has / how is / how does",
"i'd": "i would", # "I had / I would",
"i'd've": "i would have",
"i'll": "i will", # "I shall / I will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would", # "it had / it would",
"it'll": "it will", # "it shall / it will",
"it's": "it is", # "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mustn't": "must not",
"needn't": "need not",
"o'clock": "of the clock",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would", # "she had / she would",
"she'd've": "she would have",
"she'll": "she will", # "she shall / she will",
"she's": "she is", # "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"that's": "that is", # "that has / that is",
"there's": "there is", # "there has / there is",
"they'd": "they would", # "they had / they would",
"they'll": "they will", # "they shall / they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would", # "we had / we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what's": "what is", # "what has / what is",
"when've": "when have",
"where'd": "where did",
"where's": "where is", # "where has / where is",
"where've": "where have",
"who'll": "who will", # "who shall / who will",
"who's": "who is", # "who has / who is",
"who've": "who have",
"why's": "why is", # "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"would've": "would have",
"wouldn't": "would not",
"y'all": "you all",
"you'd": "you would", # "you had / you would",
"you'll": "you will", # "you shall / you will",
"you're": "you are",
"you've": "you have"
}

def process_sentence(sentence_array, purge_methods):
    for method in purge_methods:
        sentence_array = method(sentence_array)
    return sentence_array

def remove_hashtag(before):
    if len(before) == 0:
        return ""
    
    if before[0] == "#":
        return before[1:]
    return before

def remove_contractions(before):
    if before in contractions:
        return contractions[before]
    return before

def remove_end_of_line(before):
    if len(before) == 0:
        return ""
    
    if before[-1] == '\n':
        return before[:-1]
    return before

def words_to_tags(before):
    if before == "<url>":
        return "[UNK]"
    elif before == "xox":
        return "kiss"
    elif before == "<user>":
        return "alice"
    return before

def to_vec(lmt_wise_method):
    return np.vectorize(lmt_wise_method)

# Standard pipeline
preproc_pipeline = [
    to_vec(remove_end_of_line), 
    to_vec(remove_hashtag), 
    to_vec(remove_contractions), 
    to_vec(words_to_tags)
]