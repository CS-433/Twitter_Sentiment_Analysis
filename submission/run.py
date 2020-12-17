from fast_text.fasttext_intermediate import generate_intermediate as gen_fasttext
from tfidf.tfidf_intermediate import generate_intermediate as gen_tfidf
from roberta.roberta_intermediate import generate_intermediate as gen_roberta
import stage2_model as mod
import xgboost as xgb
import wget
import os

root = 'data/'
os.makedirs(root, exist_ok=True)
step = 1


print("Step {:d}.\tCalling tfidf".format(step))
tfidf_filename = root + 'tfidf_intermediate.csv'
gen_tfidf(tfidf_filename)
step += 1


print("Step {:d}.\tLoading GloVe".format(step))
glove_predictions_url = 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBclREZ3U5ejdJT1ZqcU01c2Z3VjkwVEhMaXkxZFE_ZT01QXVr/root/content'
glove_filename = root + 'glove_intermediate.csv'
wget.download(glove_predictions_url, glove_filename)
step += 1

print("Step {:d}.\tCalling RoBERTa".format(step))
roberta_filename = root + 'roberta_intermediate.csv'
#gen_roberta(roberta_filename)
step += 1

print("Step {:d}.\tCalling fastText".format(step))
fasttext_filename = root + 'fasttext_intermediate.csv'
gen_fasttext(fasttext_filename)
step += 1

print("Step {:d}.\tLoading computed intermediate results".format(step))
scores = mod.load_all_intermediate(fasttext_filename, tfidf_filename, roberta_filename, glove_filename)
step += 1

print("Step {:d}.\Loading pre-trained XGBoost model".format(step))
xgb_filename = root + 'xgb_model.json'
xgb_clf = xgb.XGBClassifier(objective ='binary:hinge', booster = 'dart', colsample_bytree = 1, learning_rate = 0.3,
                max_depth = 10, alpha = 1, n_estimators = 1, use_label_encoder=False, tree_method = 'exact', 
                         num_parallel_tree = 4)
                         
xgb_clf.load_model(xgb_filename)
step += 1

print("Step {:d}.\tSaving predictions generated with pre-trained model".format(step))
predictions = xgb_clf.predict(scores)
mod.save_pred('XGBoost_submission.csv', predictions)
print('Done')
