# uw-fsdl-bert-transfer-translation
Final project for the UW Full Stack Deep Learning course.

## Project goal
We will try to reproduce **transfer learning** in a sequence-to-sequence NMT model.

## Methodology
We will attach a pre-trained BERT model to an untrained transformer decoder, and train the combined model to translate English to French using the [EuroParl parallel corpus](http://www.statmt.org/europarl/). We will train another copy of this model to translate English to Spanish.  Finally, we will retrain the English to French model on the English to Spanish task and vice-versa, and compare the loss curves to the first set of trainings.  Our hypothesis is that the loss will decrease more quickly in the second set of trainings.

## Requirements
```
pip install sklearn tensorflow transformers
```

You will also need significant hard drive space - at least 540 MB for the BERT model and 1590 MB for the English/French and English/Spanish data.

## Set-up
There are so many ways to get and process data that this step almost becomes hard. TensorFlow has pre-built datasets, Keras has utilities for retrieving and extracting raw data files, Sergey has custom code for loading EMNIST data, etc.  In the end we built our own data pre-processing routine using Keras library functions.

The BERT model we used is huge, with over 109 million parameters. To speed up the training and avoid wrecking the pre-trained weights, we froze these parameters and stopped gradients from reaching the BERT model during backpropagation. This caused our model to have around 119 million parameters of which around 9 million were trainable.

Before attempting the French and Spanish translation tasks, we tuned the decoder hyperparameters on an English to English task until we saw the loss drop very close to zero. This ensured that our decoder had sufficient representational power to interpret BERT embeddings.

## Findings
**TODO** Plot the loss curve for all four tasks

**TODO** Decide whether transfer learning was observed

**TODO** Report BLEU scores and comment on quality of translations
