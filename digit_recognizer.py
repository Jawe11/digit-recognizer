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

train_data = train_data / 255.0 #conversion to floating point
test_data = test / 255.0

#code zum darstellen eines testbildes 

index_testbild = 676
testdaten = train_data.iloc[index_testbild]
testbild = testdaten.values.reshape(28,28)

plt.imshow(testbild, cmap=plt.cm.binary) #testbild in SW darstellen
plt.title("Angezeigte Zahl:" + str(train_label[index_testbild]))
plt.show()

# Aufbau Keras Modell

model = keras.Sequential()