import tensorflow as tf
from transformers import *
from sklearn.model_selection import train_test_split

import io
import os
import time

from CombinedBertTransformerModel import *

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
def load_dataset_new(language_path, tokenizer, num_examples=None, max_tokens=500):
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

num_examples = 300
max_tokens = 50
tokenizer = instantiate_tokenizer()
input_tensor_new, masks, segments = load_dataset_new(path_to_fr_en_en_file, tokenizer, num_examples, max_tokens)
target_tensor_new, _, _ = load_dataset_new(path_to_fr_en_fr_file, tokenizer, num_examples, max_tokens)

# Split the data into training and validation sets.  No test set for now since we're just experimenting.
input_tensor_train_new, input_tensor_val_new, target_tensor_train_new, target_tensor_val_new = train_test_split(input_tensor_new, target_tensor_new, test_size=0.2)

# Do some printing to show that the processing worked.
def convert_new(tokenizer, tensor):
    for t in tensor:
        if t != 0:
            print ("%d ----> %s" % (t, tokenizer.ids_to_tokens[t]))

print("ID to token mapping for first training example (input)")
convert_new(tokenizer, input_tensor_train_new[0])
print()
print("ID to token mapping for first training example (target)")
convert_new(tokenizer, target_tensor_train_new[0])

BATCH_SIZE_NEW = 64
steps_per_epoch_new = len(input_tensor_train_new) // BATCH_SIZE_NEW
embedding_dim_new = 32
units_new = 32
vocab_size_new = len(tokenizer.vocab)

train_dataset_new = tf.data.Dataset.from_tensor_slices((input_tensor_train_new, target_tensor_train_new)).shuffle(len(input_tensor_train_new)).batch(BATCH_SIZE_NEW, drop_remainder=True)
validation_dataset_new = tf.data.Dataset.from_tensor_slices((input_tensor_val_new, target_tensor_val_new)).shuffle(len(input_tensor_val_new)).batch(BATCH_SIZE_NEW, drop_remainder=True)

###################################
###      MODEL PREPARATION      ###
###################################

# Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model_new = CombinedBertTransformerModel(
    input_shape=train_dataset_new.as_numpy_iterator().next()[0].shape,
    vocab_size=vocab_size_new,
    num_layers=2,
    units=32,
    d_model=32,
    num_heads=2,
    dropout=0.2,
    padding_label=0
)
model_new.compile(optimizer=optimizer, loss=loss, metrics=[metric])
model_new.summary()

# Train and evaluate the model using tf.keras.Model.fit()
# TODO This doesn't work yet.
history = model_new.fit(
    train_dataset_new,
    epochs=2,
    steps_per_epoch=115,
    validation_data=validation_dataset_new,
    validation_steps=7
)

