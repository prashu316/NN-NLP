import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import losses
import matplotlib.pyplot as plt

#load dataset using pandas
dataset_dir = os.path.join(os.getcwd(), 'train.csv')
dataframe = pd.read_csv(dataset_dir)

print(dataframe.head())

#pop the unwanted column (id)
train_ds=dataframe.copy()
train_ds.pop('id')

print(train_ds.head())

#load test dataset
dataset_dir1 = os.path.join(os.getcwd(), 'test.csv')
dataframe1 = pd.read_csv(dataset_dir1)

test_ds=dataframe1.copy()


#create train validation split
train, val = train_test_split(dataframe, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')





#function to convert numpy object to tensor
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  dataframe.pop('id') #pop unwanted column
  labels = dataframe.pop('label')
  tweet=dataframe.pop('tweet')
  ds = tf.data.Dataset.from_tensor_slices((tweet, labels))

#shuffle the data and batch it
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds


#separate function to convert test dataset to tensor and batch it
def df_to_dataset1(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  dataframe.pop('id')

  tweet=dataframe.pop('tweet')
  ds = tf.data.Dataset.from_tensor_slices(tweet)


  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds


batch_size = 10



train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, batch_size=batch_size)
test_ds = df_to_dataset1(test_ds, batch_size=batch_size)




#printing sample reviews
for text_batch, label_batch in train_ds.take(1):
  for i in range(3):
    print("Review", text_batch.numpy()[i])
    print("Label", label_batch.numpy()[i])




max_features = 5000
sequence_length = 100

#create a text vectorisation layer
vectorize_layer = layers.TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

print(train_ds)

#use a features only tensor in order to build the vocabulary use adapt function
train_text = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

print(train_text)

#function to vectorize text
def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

def vectorize_text1(text):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text)

'''
text_batch, label_batch = next(iter(train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Vectorized review", vectorize_text(first_review, first_label))
'''

#printing out value of integer in array
'''
print("1287 ---> ",vectorize_layer.get_vocabulary()[287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))
'''

#apply the vectorize layer on all inputs
train_ds = train_ds.map(vectorize_text)
val_ds = val_ds.map(vectorize_text)
test_ds = test_ds.map(vectorize_text1)

#making reading of input file and making model more efficient
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#creating the model
embedding_dim = 16
latent_dim=128
model = tf.keras.Sequential([

  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.LSTM(128),
  layers.Dropout(0.2),
  layers.Dense(1)
])

print(model.summary())

#loss function
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

#training the model
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
)

loss, accuracy = model.evaluate(val_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

#the program stores a dictionary of everything that happened during training
history_dict = history.history
print(history_dict.keys())

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()


#exporting model to work on raw strings, vectorize layer inside creating the model

export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)


