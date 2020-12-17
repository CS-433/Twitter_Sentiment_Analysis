from fast_text.fasttext_intermediate import generate_intermediate as gen_fasttext
from tfidf.tfidf_intermediate import generate_intermediate as gen_tfidf
import stage2_model as mod
import os
root = 'data/'
os.makedirs(root, exist_ok=True)

step = 1
print("Step {:d}.\tCalling fastText".format(step))
fasttext_filename = 'data/fasttext_intermediate.csv'
gen_fasttext(fasttext_filename)
step += 1

print("Step {:d}.\tCalling tfidf".format(step))
tfidf_filename = 'data/tfidf_intermediate.csv'
gen_fasttext(tfidf_filename)
step += 1

print("Step {:d}.\tLoading intermediate results".format(step))
scores = mod.load_all_intermediate(fasttext_filename, tfidf_filename)
step += 1

print("Step {:d}.\tGenerating predictions with pre-trained model".format(step))
