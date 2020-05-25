import numpy as np
import tensorflow as tf
from transformers import *

from decoder import *
from masking import *

def CombinedBertTransformerModel(
    max_tokens: int,
    vocab_size: int,
    num_layers: int,
    units: int,
    d_model: int,
    num_heads: int,
    dropout: float,
    padding_label: int = 0,
) -> tf.keras.Model:

    bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    # Freeze the weights and biases in the BERT model.
    for layer in bert_model.layers:
        layer.trainable = False

    tokenized_input_sentence = tf.keras.Input(shape=(max_tokens,), name="tokenized_input_sentence", dtype=tf.int32)
    bert_outputs = bert_model(tokenized_input_sentence)[0]

    tokenized_output_sentence = tf.keras.Input(shape=(max_tokens,), name="tokenized_output_sentence", dtype=tf.int32)

    # Mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(
        lambda x: create_look_ahead_mask(x, padding_label=padding_label),
        output_shape=(1, None, None),
        name="look_ahead_mask",
    )(tokenized_output_sentence)

    dec_outputs = decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        d_enc_outputs=bert_model.output_shape[1][1],
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[tokenized_output_sentence, bert_outputs, look_ahead_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[tokenized_input_sentence, tokenized_output_sentence], outputs=outputs)
