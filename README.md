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

We trained on only a small subset of the EuroParl corpus (400 training batches, 100 validation batches, 64 sentence pairs per batch). Even so, each training took ~1 hour on a GTX 1080 Ti. Final accuracy might be improved by repeating this experiment with more data.

## Findings
We observed significant transfer learning in both trials. The loss and accuracy plateaued around the same values regardless of whether the models were pre-trained, but the pre-trained models were able to reach this plateau sooner.
 - 10 epochs of pre-training on Spanish followed by 1 epoch training on French were slightly more effective than 5 epochs training on French from scratch.
 - 10 epochs of pre-training on French followed by 1 epoch training on Spanish were slightly more effective than 6 epochs training on Spanish from scratch.
![Plots of accuracy and loss by epoch](https://github.com/tuanh118/uw-fsdl-bert-transfer-translation/raw/master/plots/Plots.png)

**TODO** Report BLEU scores and comment on quality of translations
