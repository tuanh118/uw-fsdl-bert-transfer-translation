import tensorflow as tf
from util import *

def run_transformer_inference(
    model: tf.keras.models.Model, input: tf.Tensor, max_length: int, start_label: int, end_label: int
) -> tf.Tensor:
    target = tf.expand_dims([start_label], 0)

    for i in range(max_length):
        predictions = model(inputs=[tf.expand_dims(input, 0), target], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, end_label):
            break

        # concatenated the predicted_id to the output which is given to the decoder as input
        target = tf.concat([target, predicted_id], axis=-1)

    return tf.squeeze(target, axis=0)

tokenizer = instantiate_tokenizer()

# Load your model and data here. Data must be tokenized.
model = None
target_tensor = None
input_tensor = None

# Disable TensorFlow warnings due to tensor dimension mismatch.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

original_file = open('original.txt', 'w')
translated_file = open('translated.txt', 'w')
reference_translated_file = open('reference_translated.txt', 'w')

for ix, input in enumerate(input_tensor):
    original_sentence = tokenizer.decode(input, skip_special_tokens=True)
    translated_tokens = run_transformer_inference(model, input=input, max_length=max_tokens, start_label=tokenizer.cls_token_id, end_label=tokenizer.sep_token_id)
    translated_sentence = tokenizer.decode(translated_tokens, skip_special_tokens=True)
    reference_translated_sentence = tokenizer.decode(target_tensor[ix], skip_special_tokens=True)
    print('Original sentence: ' + original_sentence)
    print('Translated sentence: ' + translated_sentence)
    print('Reference translation: ' + reference_translated_sentence)
    original_file.write(original_sentence + '\n')
    translated_file.write(translated_sentence + '\n')
    reference_translated_file.write(reference_translated_sentence + '\n')
