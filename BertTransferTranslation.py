import os
import tarfile
import torch

from BertTransformerModel import BertTransformerModel

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

fr_en_corpus = load_parallel_corpus("fr", "en")
es_en_corpus = load_parallel_corpus("es", "en")

# Print the first line of each corpus to prove the reading worked.
print(fr_en_corpus[0][0])
print(fr_en_corpus[1][0])
print(es_en_corpus[0][0])
print(es_en_corpus[1][0])

# Run one sentence through BERT to prove encoding works.
model = BertTransformerModel()
print(model.forward(fr_en_corpus[1][0]))
