import tensorflow as tf   # Tensorflow laden
from tensorflow import keras #Keras laden
import pandas as pd         # Pandas zur Datenanalyse
import matplotlib.pyplot as plt #Pyplot zur graphischen Darstellung

train = pd.read_csv("C:/Users/janwe/Documents/GitHub/digit-recognizer/train.csv",sep=",") #Pfad zu CSV Files
test = pd.read_csv("C:/Users/janwe/Documents/GitHub/digit-recognizer/test.csv",sep=",")

# print(train.head())
# print(test.head())

train_label = train["label"]
train_data = (train.drop("label", axis = 1))

#print(train_data.shape) -> (42000, 784)

train_data = train_data / 255.0 #conversion to floating point
test_data = test / 255.0

#code zum darstellen eines testbildes 


index_testbild = 676
testdaten = train_data.iloc[index_testbild]
testbild = testdaten.values.reshape(28,28)

plt.imshow(testbild, cmap=plt.cm.binary) #testbild in SW darstellen
plt.title("Angezeigte Zahl:" + str(train_label[index_testbild]))
plt.show()

#umformen aller Daten in 2D Matrix 28x28
train_data = train_data.values.reshape(-1,28,28)
test_data = test_data.values.reshape(-1,28,28)
#print(train_data.shape) -> (42000, 28, 28)


# Aufbau Keras Modell
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)), #Flatten nötig da schon eindimensional?
  tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax") #"softmax baked into last layer" google rät davon ab
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#training keras model
model.fit(train_data, train_label, epochs=5)


