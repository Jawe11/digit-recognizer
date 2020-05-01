import tensorflow as tf   # Tensorflow laden
import pandas as pd         # Pandas zur Datenanalyse
import matplotlib.pyplot as plt #Pyplot zur graphischen Darstellung

train = pd.read_csv("C:/Users/janwe/Documents/GitHub/digit-recognizer/train.csv",sep=",") #Pfad zu CSV Files
test = pd.read_csv("C:/Users/janwe/Documents/GitHub/digit-recognizer/test.csv",sep=",")

# print(train.head())
# print(test.head())

train_label = train["label"]
# print(train_label)

train_data = (train.drop("label", axis = 1))

#code zum darstellen einer zahl 

index_testbild = 676
testdaten = train_data.iloc[index_testbild]
testbild = testdaten.values.reshape(28,28)

plt.imshow(testbild)
plt.title("Angezeigte Zahl:" + str(train_label[index_testbild]))
plt.show()

print(testbild)