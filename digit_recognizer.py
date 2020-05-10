import tensorflow as tf   # Tensorflow laden
from tensorflow import keras #Keras laden
import pandas as pd         # Pandas zur Datenanalyse
import matplotlib.pyplot as plt #Pyplot zur graphischen Darstellung
import numpy as np
from random import randint


mnist = tf.keras.datasets.mnist
(train_data, train_label), (test_data, test_label) = mnist.load_data()

train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)


train_data = train_data / 255.0 #conversion to floating point
test_data = test_data / 255.0


# Aufbau Keras Modell

#filters is the number of desired feature maps.
#kernel_size is the size of the convolution kernel. A single number 5 means a 5x5 convolution.
#strides the new layer maps will have a size equal to the previous layer maps divided by strides. Leaving this blank results in strides=1.
#padding is either 'same' or 'valid'. Leaving this blank results in padding='valid'. If padding is 'valid' then the size of the new layer maps is reduced by kernel_size-1. For example, if you perform a 5x5 convolution on a 28x28 image (map) with padding='valid', then the next layer has maps of size 24x24. If padding is 'same', then the size isn't reduced.
#activation is applied during forward propagation. Leaving this blank results in no activation.

# 3 Layers Convolutional am besten, jedoch 2 annähernd genausogut
# 
model = tf.keras.models.Sequential([
  
  tf.keras.layers.Conv2D(32, kernel_size =3, input_shape= input_shape, activation='relu'), #erstes Conv2D mit 32 feature Maps
  tf.keras.layers.Conv2D(32, kernel_size =3, input_shape= input_shape, activation='relu'), #zwei 3x3 Conv2D statt 1x 5x5
  tf.keras.layers.BatchNormalization(),# Normalisierung der Aktivierungsfunktion zwischen den mini batches
  tf.keras.layers.Conv2D(64,kernel_size = 5, strides = 2, padding = "same", activation='relu'), #ersetzt MaxPool2D
  tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Dropout(0.4),#ideal zwischen 0,2 und 0,4

  tf.keras.layers.Conv2D(64,kernel_size = 3, activation='relu'), #zweites Conv2D mit 64 feature Maps 
  tf.keras.layers.Conv2D(64,kernel_size = 3, activation='relu'), #zwei 3x3 Conv2D statt 1x 5x5
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(64,kernel_size = 5, strides = 2, padding = "same", activation='relu'),
  tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Dropout(0.4),

	tf.keras.layers.Flatten(),

	tf.keras.layers.Dense(128, activation='relu'), #ab 128 Neuronen keine signifikante Verbesserung
  tf.keras.layers.Dropout(0.4),

	tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


#training keras model
epochs = 20 #Anzahl an Trainings Iterationen
history = model.fit(train_data, train_label,validation_data=(test_data,test_label), epochs = epochs)



#Auswertung des Trainingsfortschritts
acc = history.history['accuracy']
loss=history.history['loss']
val_acc = history.history['val_accuracy']
val_loss=history.history['val_loss']


epochs_range = range(epochs)

# list all data in history
# print(history.history.keys()) -> Anzeigen aller verfügbaren history Datensätze

plt.figure(figsize=(8, 8))
plt.subplot(1, 2,1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training & Validation Accuracy')
plt.subplot(1, 2,2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training & Validation Loss')
plt.show()


#Bild vorhersagen
vorhersage = model.predict(test_data)





#code zum darstellen zufälliger testbilder
fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 4

train_data = train_data.reshape(train_data.shape[0], 28, 28)
test_data = test_data.reshape(test_data.shape[0], 28, 28)

for i in range(1,17):
    index_testbild = randint(0, 10000) #Zufällige Auswahl eines anzuzeigenden Testbildes aus test_data
    wahrscheinlichkeit_testbild = np.amax(vorhersage[index_testbild]) #Höchste Wahrscheinlichkeit auf ExitLayer
    wahrscheinlichkeit_testbild = wahrscheinlichkeit_testbild*100 #Zu Prozent wandeln
    vorhersage_testbild = np.argmax(vorhersage[index_testbild]) #Neuron auf Exitlayer mit höchster Wahrscheinlichkeit auswählen
    fig.add_subplot(rows, columns, i) 
    plt.imshow(test_data[index_testbild], cmap="Greys") #testbild in SW darstellen
    plt.title("Ermittelte Zahl: " + str(vorhersage_testbild))
    plt.xlabel(str(wahrscheinlichkeit_testbild.round(2))+" % ")#Anzeige derWahrscheinlichkeit der Vorhersage auf 2 Nachkommastellen gerundet
plt.subplots_adjust(hspace=1.8, wspace = 1.8) #Abstand der Plots
plt.show()



