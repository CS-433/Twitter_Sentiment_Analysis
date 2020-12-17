from fast_text.fasttext_intermediate import generate_intermediate as gen_fasttext
from tfidf.tfidf_intermediate import generate_intermediate as gen_tfidf
from roberta.roberta_intermediate import generate_intermediate as gen_roberta
import stage2_model as mod
import wget
import os

root = 'data/'
os.makedirs(root, exist_ok=True)


print("Step {:d}.\tLoading GloVe".format(step))
glove_predictions_url = 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBclREZ3U5ejdJT1ZqcU01c2Z3VjkwVEhMaXkxZFE_ZT01QXVr/root/content'
glove_filename = root + 'glove_intermediate.csv'
wget.download(glove_predictions_url, glove_filename)
step += 1


step = 1
print("Step {:d}.\tCalling fastText".format(step))
fasttext_filename = root + 'fasttext_intermediate.csv'
gen_fasttext(fasttext_filename)
step += 1

print("Step {:d}.\tCalling tfidf".format(step))
tfidf_filename = root + 'tfidf_intermediate.csv'
gen_tfidf(tfidf_filename)
step += 1

print("Step {:d}.\tCalling RoBERTa".format(step))
roberta_filename = root + 'roberta_intermediate.csv'
gen_roberta(roberta_filename)
step += 1

print("Step {:d}.\tLoading intermediate results".format(step))
scores = mod.load_all_intermediate(fasttext_filename, tfidf_filename, glove_filename, roberta_filename)
step += 1

print("Step {:d}.\tGenerating predictions with pre-trained model".format(step))
