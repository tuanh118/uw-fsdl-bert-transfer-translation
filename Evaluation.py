import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize

def evaluate_translation(true_filename, prediction_filename):

    tokenizer = nltk.RegexpTokenizer(r"\w+")

    with open(true_filename, 'r', encoding='utf-8') as truefile:
        references = [ [tokenizer.tokenize(line)] for line in truefile.read().splitlines() ]

    with open(prediction_filename, 'r', encoding='utf-8') as predfile:
        hypotheses = [ tokenizer.tokenize(line) for line in predfile.read().splitlines() ]

    return corpus_bleu(references, hypotheses)

print(evaluate_translation('y_test_true.en-fr', 'y_test_pred_fairseq.en-fr'))
# 0.3738055305707718