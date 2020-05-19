import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.tokenize import word_tokenize

def evaluate_translation(true_filename, prediction_filename):

    tokenizer = nltk.RegexpTokenizer(r"\w+")

    with open(true_filename, 'r', encoding='utf-8') as truefile:
        references = [ [tokenizer.tokenize(line)] for line in truefile.read().splitlines() ]

    # print('references', references)

    with open(prediction_filename, 'r', encoding='utf-8') as predfile:
        hypotheses = [ tokenizer.tokenize(line) for line in predfile.read().splitlines() ]

    # print('hypotheses', hypotheses)

    return corpus_bleu(references, hypotheses)

print(evaluate_translation('y_test_true.fr-en.txt', 'y_test_pred.fr-en.txt'))
# 0.4302023469880712