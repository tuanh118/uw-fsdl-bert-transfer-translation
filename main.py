import tensorflow as tf
from transformers import *
from sklearn.model_selection import train_test_split

import io
import os
import time

from CombinedBertTransformerModel import *
from DatasetSequence import *
from functools import partial
from util import *

###################################
###        DATA RETRIEVAL       ###
###################################

# Download the EuroParl French-English corpus.
path_to_fr_en_tar = tf.keras.utils.get_file('fr-en.tgz', origin='https://www.statmt.org/europarl/v7/fr-en.tgz', extract=True)
path_to_fr_en_en_file = os.path.dirname(path_to_fr_en_tar) + "/europarl-v7.fr-en.en"
path_to_fr_en_fr_file = os.path.dirname(path_to_fr_en_tar) + "/europarl-v7.fr-en.fr"

###################################
###       DATA PROCESSING       ###
###################################

# Sets up a BERT tokenizer.
def instantiate_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

# Given a path to a text file, load and tokenize each line using the provided tokenizer, then convert each token to an ID and pad all lines to have length max_tokens.
def load_dataset(language_path, tokenizer, num_examples=None, max_tokens=500):
    # Read the data.
    lines = io.open(language_path, encoding='UTF-8').read().strip().splitlines()[:num_examples]

    # Tokenize and add the special start token.
    tokenized_lines = [ ['[CLS]'] + tokenizer.tokenize(line)[:max_tokens-1] + ['[SEP]'] for line in lines ]
    
    # Convert tokens to IDs.
    ids = [ tokenizer.convert_tokens_to_ids(tokenized_line) for tokenized_line in tokenized_lines ]

    # Generate padding masks and segment IDs. These have the same length as the ID sequences after padding.
    # Padding mask is 1 where there is an actual ID and 0 where there is padding. Segment ID is always 0.
    masks = [ [1] * len(tokenized_line) for tokenized_line in tokenized_lines ]
    segments = [ [] for tokenized_line in tokenized_lines ]

    # Pad all ID sequences to the maximum length with zeroes.
    ids = tf.keras.preprocessing.sequence.pad_sequences(ids, maxlen=max_tokens, truncating="post", padding="post", dtype="int")
    masks = tf.keras.preprocessing.sequence.pad_sequences(masks, maxlen=max_tokens, truncating="post", padding="post", dtype="int")
    segments = tf.keras.preprocessing.sequence.pad_sequences(segments, maxlen=max_tokens, truncating="post", padding="post", dtype="int")

    return ids, masks, segments

BATCH_SIZE = 64
d_model = 32
num_examples = BATCH_SIZE * 5
max_tokens = 200
tokenizer = instantiate_tokenizer()
vocab_size = len(tokenizer.vocab)

input_tensor, masks, segments = load_dataset(path_to_fr_en_en_file, tokenizer, num_examples, max_tokens)
target_tensor, _, _ = load_dataset(path_to_fr_en_fr_file, tokenizer, num_examples, max_tokens)

# Split the data into training and validation sets.  No test set for now since we're just experimenting.
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# Do some printing to show that the processing worked.
def convert(tokenizer, tensor):
    for t in tensor:
        if t != 0:
            print ("%d ----> %s" % (t, tokenizer.ids_to_tokens[t]))

print("ID to token mapping for first training example (input)")
convert(tokenizer, input_tensor_train[0])
print()
print("ID to token mapping for first training example (target)")
convert(tokenizer, target_tensor_train[0])

def format_batch(x, y):
    """
    Inputs are x and y up to the last character.
    Outputs are y from first character (shifted).
    """
    return [x, y[:, :-1]], y[:, 1:]

train_dataset = DatasetSequence(input_tensor_train, target_tensor_train, batch_size=BATCH_SIZE, format_fn=format_batch)
validation_dataset = DatasetSequence(input_tensor_val, target_tensor_val, batch_size=BATCH_SIZE, format_fn=format_batch)

###################################
###      MODEL PREPARATION      ###
###################################

# Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
learning_rate = CustomSchedule(d_model=d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

loss = partial(sparse_categorical_crossentropy_ignoring_padding, padding_label=0)
loss.__name__ = 'loss'
accuracy = partial(sparse_categorical_accuracy_ignoring_padding, padding_label=0)
accuracy.__name__ = 'accuracy'

model = CombinedBertTransformerModel(
    max_tokens=max_tokens,
    vocab_size=vocab_size,
    num_layers=2,
    units=32,
    d_model=d_model,
    num_heads=2,
    dropout=0,
    padding_label=0
)
model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])
model.summary()

# Train and evaluate the model using tf.keras.Model.fit()
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    use_multiprocessing=False,
    workers=1,
    shuffle=True,
    epochs=10
)
