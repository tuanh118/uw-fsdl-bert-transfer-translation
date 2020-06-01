import pickle
import matplotlib.pyplot as plt

# Replace these with the paths to your saved model histories.
history_french_from_scratch = pickle.load(open("history20200530192526", "rb"))
history_french_pretrained_on_spanish = pickle.load(open("history20200531131043", "rb"))
history_spanish_from_scratch = pickle.load(open("history_fr_20200531151710", "rb"))
history_spanish_pretrained_on_french = pickle.load(open("history_fr_es_20200531173917", "rb"))
epochs = list(range(1, 11))

# Plot validation set accuracy for both French models.
plt.plot(epochs, history_french_from_scratch['val_accuracy'])
plt.plot(epochs, history_french_pretrained_on_spanish['val_accuracy'])
plt.title('EN to FR: Validation set accuracy by epoch')
plt.axis([1, 10, 0, 1])
plt.ylabel('EN -> FR validation set accuracy')
plt.xlabel('Epoch')
plt.legend(['From scratch', 'Pretrained on Spanish'])
plt.show()

# Plot validation set loss for both French models.
plt.plot(epochs, history_french_from_scratch['val_loss'])
plt.plot(epochs, history_french_pretrained_on_spanish['val_loss'])
plt.title('EN to FR: Validation set loss by epoch')
plt.axis([1, 10, 0, 2])
plt.ylabel('EN -> FR validation set loss')
plt.xlabel('Epoch')
plt.legend(['From scratch', 'Pretrained on Spanish'])
plt.show()

# Plot validation set accuracy for both Spanish models.
plt.plot(epochs, history_spanish_from_scratch['val_accuracy'])
plt.plot(epochs, history_spanish_pretrained_on_french['val_accuracy'])
plt.title('EN to ES: Validation set accuracy by epoch')
plt.axis([1, 10, 0, 1])
plt.ylabel('EN -> ES validation set accuracy')
plt.xlabel('Epoch')
plt.legend(['From scratch', 'Pretrained on French'])
plt.show()

# Plot validation set loss for both Spanish models.
plt.plot(epochs, history_spanish_from_scratch['val_loss'])
plt.plot(epochs, history_spanish_pretrained_on_french['val_loss'])
plt.title('EN to ES: Validation set loss by epoch')
plt.axis([1, 10, 0, 2])
plt.ylabel('EN -> ES validation set loss')
plt.xlabel('Epoch')
plt.legend(['From scratch', 'Pretrained on French'])
plt.show()