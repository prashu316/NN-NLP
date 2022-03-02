import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np
import os

#open sample text used as dataset for training
file = open("pred.txt", "r", encoding="utf8")
lines = []

#store each line separately
for i in file:
    lines.append(i)

'''
print("The First Line: ", lines[0])
print("The Last Line: ", lines[-1])
'''

data = ""

#store each word separately, the training set consists of words rather than sentences
for i in lines:
    data = ' '.join(lines)

#remove end lines
data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '')
#print(data[:360])


import string

#remove punctuations
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
new_data = data.translate(translator)

#print(new_data[:500])

#store all unique words in z
z = []

for i in data.split():
    if i not in z:
        z.append(i)

data = ' '.join(z)
#print(data[:500])

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# saving the tokenizer for predict function using pickle
pickle.dump(tokenizer, open('tokenizer1.pkl', 'wb'))

#convert words to number indices
sequence_data = tokenizer.texts_to_sequences([data])[0]
print(sequence_data[:10])


#length of vocabulary
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

#create an array sequences which stores every 2 consecutive words
sequences = []

for i in range(1, len(sequence_data)):
    words = sequence_data[i - 1:i + 1]
    sequences.append(words)

print("The Length of sequences are: ", len(sequences))
sequences = np.array(sequences)
#print(sequences[:10])

#create feature and target vectors
X = []
y = []

for i in sequences:
    X.append(i[0])
    y.append(i[1])

X = np.array(X)
y = np.array(y)
print(y.shape)

#one hot encode target vector as values can only be 0 and 1
y = to_categorical(y, num_classes=vocab_size)
print(y[:5])
print(y.shape)
BATCH_SIZE = 64
BUFFER_SIZE=len(X)

#function to convert numpy object to tensor
dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)

print(dataset)

#creating the model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(vocab_size, activation="softmax"))

from tensorflow import keras
from keras.utils.vis_utils import plot_model

model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001))

model.fit(dataset, epochs=20, batch_size=64)

