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
### Transfer learning
We observed significant transfer learning in both trials. The loss and accuracy plateaued around the same values regardless of whether the models were pre-trained, but the pre-trained models were able to reach this plateau sooner.
 - 10 epochs of pre-training on Spanish followed by 1 epoch training on French were slightly more effective than 5 epochs training on French from scratch.
 - 10 epochs of pre-training on French followed by 1 epoch training on Spanish were slightly more effective than 6 epochs training on Spanish from scratch.
![Plots of accuracy and loss by epoch](https://github.com/tuanh118/uw-fsdl-bert-transfer-translation/raw/master/plots/Plots.png)

### Translation quality
We used each trained model to translate 300 hold-out sentences using greedy search, and compared these to the reference translations. Many of these sentences had complex grammatical structure due to the nature of the corpori used.

The few several words of each translation are often of good quality in terms of grammar and semantic meaning, but quality declines as the length of the translation increases, likely due to our use of greedy search. The models also have trouble with numbers, sometimes altering numbers slightly in translation.  Using beam search with beam size K>1 likely would have produced higher-quality translations, but we did not attempt this.

Some particular examples (English to French):

 - Original sentence: mr president, after the voting, i would like to raise a point of order concerning the texts adopted yesterday.
   - Translated sentence: monsieur le president, apres le vote, apres le vote de clos concernant les textes de vote.
   - Reference translation: monsieur le president, apres le vote, je voudrais encore soulever une motion de procedure concernant les textes adoptes hier.

 - Original sentence: ( the sitting was closed at 10. 50 a. m. )
   - Translated sentence: ( la seance est levee a 11h50 )
   - Reference translation: ( la seance est levee a 10h50 )

## Future directions
We could have improved accuracy by trying any of the following:
 - **Using more training data** (we used less than 2% of the EuroParl corpus).
 - Training the models for longer (we did not train long enough to observe overfitting).
 - Finer hyperparameter tuning of the decoder.
 - Unfreezing the BERT parameters once the loss plateaued.
 - Replacing greedy search with beam search for inference.
 - Following tips from the paper [Training Tips for the Transformer Model](https://ufal.mff.cuni.cz/pbml/110/art-popel-bojar.pdf).
