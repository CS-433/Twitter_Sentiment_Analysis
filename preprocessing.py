import numpy as np
import wordsegment
from wordsegment import load, segment
load()
from spellchecker import SpellChecker
spell = SpellChecker()

import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps=PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

import re

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
"you've": "you have",
###################################################################
## Also, many tweets have the same contractions without accents  ##
###################################################################
"aint": "be not", # "am not / are not / is not / has not / have not",
"arent": "be not", # "are not / am not",
"cant": "cannot",
"cantve": "cannot have",
"cause": "because",
"couldve": "could have",
"couldnt": "could not",
"couldntve": "could not have",
"didnt": "did not",
"doesnt": "does not",
"dont": "do not",
"hadnt": "had not",
"hadntve": "had not have",
"hasnt": "has not",
"havent": "have not",
"hed": "he would", # "he had / he would",
"hedve": "he would have",
"hell": "he will", # "he shall / he will",
"hes": "he is", # "he has / he is",
"howd": "how did",
"howdy": "how do you",
"howdy": "how do you",
"howll": "how will",
"hows": "how", # "how has / how is / how does",
"id": "i would", # "I had / I would",
"idve": "i would have",
"ill": "i will", # "I shall / I will",
"im": "i am",
"ive": "i have",
"isnt": "is not",
"itd": "it would", # "it had / it would",
"itll": "it will", # "it shall / it will",
"its": "it is", # "it has / it is",
"lets": "let us",
"maam": "madam",
"mustnt": "must not",
"neednt": "need not",
"oclock": "of the clock",
"oughtnt": "ought not",
"shant": "shall not",
"shant": "shall not",
"shed": "she would", # "she had / she would",
"shedve": "she would have",
"shell": "she will", # "she shall / she will",
"shes": "she is", # "she has / she is",
"shouldve": "should have",
"shouldnt": "should not",
"thats": "that is", # "that has / that is",
"theres": "there is", # "there has / there is",
"theyd": "they would", # "they had / they would",
"theyll": "they will", # "they shall / they will",
"theyre": "they are",
"theyve": "they have",
"wasnt": "was not",
"wed": "we would", # "we had / we would",
"well": "we will",
"were": "we are",
"weve": "we have",
"werent": "were not",
"whats": "what is", # "what has / what is",
"whenve": "when have",
"whered": "where did",
"wheres": "where is", # "where has / where is",
"whereve": "where have",
"wholl": "who will", # "who shall / who will",
"whos": "who is", # "who has / who is",
"whove": "who have",
"whys": "why is", # "why has / why is",
"whyve": "why have",
"willve": "will have",
"wont": "will not",
"wouldve": "would have",
"wouldnt": "would not",
"yall": "you all",
"youd": "you would", # "you had / you would",
"youll": "you will", # "you shall / you will",
"youre": "you are",
"youve": "you have"
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


def split_hashtag(before):
    if len(before) == 0:
        return ""
    
    if before[0] == "#":
        return ' '.join(segment(before))
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


def spell_correction(before):
    return spell.correction(before) 

def truncate_small_words(before , minimum_size):
    if len(before)<minimum_size:
        return ""
    return before

def remove_digits(before):
    if before.isdigit() :
        return ""
    return before

def remove_stopwords(before):
    if before in stop_words:
        return ""
    return before
    
def stemming(before):
    return ps.stem(before)

def lemmatize(before):
    return wordnet_lemmatizer.lemmatize(before)

positive_emojis = r"(([<>]?[:;=8][\-o\*]?[\)\]dDpP\}@])|([\(\[dDpP\{@][\-o\*]?[:;=8][<>]?))"
negative_emojis = r"(([<>]?[:;=8][\-o\*\']?[\(\[/\{\|\\])|([\)\]/\}\|\\][\-o\*\']?[:;=8][<>]?))"

# Elaborated from: (credit http://sentiment.christopherpotts.net/code-data/happyfuntokenizing.py)
# (
#   (
#     [<>]?
#     [:;=8]
#     [\-o\*\']?
#     [\)\]\(\[dDpP/\:\}\{@\|\\]
#   )|(
#     [\)\]\(\[dDpP/\:\}\{@\|\\]
#     [\-o\*\']?
#     [:;=8]
#     [<>]?
#   )
# )

def translate_emojis(before):
    if len(before) == 0:
        return ""

    pos = re.match(positive_emojis, before)
    neg = re.match(negative_emojis, before)

    if pos is not None:
        return "happy"
    elif neg is not None:
        return "sad"
    else:
        return before
    
def remove_repeats(text):
    """ Replace repeated letters by single letter. """
    return re.sub(r'([a-z])\1+', r'\1', text)

def to_vec(lmt_wise_method):
    return np.vectorize(lmt_wise_method)

# Standard pipeline
preproc_pipeline = [
    to_vec(remove_end_of_line),
    to_vec(split_hashtag), 
    to_vec(remove_contractions), 
    to_vec(words_to_tags),
    #to_vec(remove_digits),
    to_vec(remove_stopwords),
    to_vec(lemmatize) 
]