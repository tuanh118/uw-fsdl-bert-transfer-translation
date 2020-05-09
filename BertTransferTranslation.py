import tarfile
import torch
import os

from transformers import *

# Reads a file from the Europarl parallel corpus.
def load_parallel_corpus(language1, language2):
    BASE_URL = "http://www.statmt.org/europarl/v7/"
    FILENAME = f"{language1}-{language2}.tgz"
    INNER_FILENAMES = [ f"europarl-v7.{language1}-{language2}.{language1}", f"europarl-v7.{language1}-{language2}.{language2}" ]
    
    print(f"Locating {language1}-{language2} parallel corpus...")
    try:
        tar = tarfile.open(FILENAME, "r:gz")
    except OSError:
        print(f"Cannot open {FILENAME}; make sure it is present in the same folder as this script.")
        return

    print("Unpacking...")
    inner_files = [ tar.extractfile(inner_filename) for inner_filename in INNER_FILENAMES ]
    
    print("Reading...")
    corpori = [ inner_file.read().decode('utf-8').splitlines() for inner_file in inner_files ]
    print("Done!")

    return corpori

# Tokenizes a sentence, runs it through a pre-trained BERT model and returns the output tensor.
def encode_sentence(input):
    # This downloads the BERT model the first time it is run, which is ~430 MB and may take some time.
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    input_ids = torch.tensor([tokenizer.encode(input, add_special_tokens=True)])
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]

    return last_hidden_states

fr_en_corpus = load_parallel_corpus("fr", "en")
es_en_corpus = load_parallel_corpus("es", "en")

# Print the first line of each corpus to prove the reading worked.
print(fr_en_corpus[0][0])
print(fr_en_corpus[1][0])
print(es_en_corpus[0][0])
print(es_en_corpus[1][0])

# Run one sentence through BERT to prove encoding works.
print(encode_sentence(fr_en_corpus[1][0]))
