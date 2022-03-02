import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import string
import os
from PIL import Image
import glob
from pickle import dump, load
from tensorflow.keras.utils import to_categorical
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
import pickle
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#The dataset used is the Flickr 8k image dataset
#function to load captions
def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

filename = "image_captioning/captions.txt"
doc = load_doc(filename)
#print(doc[:500])


#function to map image id to each of its 5 captions as a dict
def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# extract filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		mapping[image_id].append(image_desc)
	return mapping

# parse the captions
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))

#check out an example
'''
print(list(descriptions.keys())[:5])
print(descriptions['1000268201_693b08cb0e'])
'''


#clean the caption data
def clean_descriptions(descriptions):

	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)

# clean descriptions
clean_descriptions(descriptions)

#print(descriptions['1000268201_693b08cb0e'])

#creating a vocabulary

# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
print('Original Vocabulary Size: %d' % len(vocabulary))

# save descriptions to file, one per line, run only once
'''
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

save_descriptions(descriptions, 'descriptions.txt')
'''



# create a list of image ids
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# load training dataset
filename = 'image_captioning/text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))

# Below path contains all the images
images = 'image_captioning/Images/'

# Create a list of all image names in the directory
img = glob.glob(images + '*.jpg')
print(len(img))

# Below file contains the names of images to be used in train data
train_images_file = 'image_captioning/text/Flickr_8k.trainImages.txt'
# Read the train image names in a set
train_images = set(open(train_images_file, 'r').read().strip().split('\n'))

# Create a list of all the training images with their full path names
train_img = []

for i in img: # img is list of full path names of all images
	if i[len(images):] in train_images: # Check if the image belongs to training set
		train_img.append(i) # Add it to the list of train images

print(len(train_img))

# Below file contains the names of images to be used in test data
test_images_file = 'image_captioning/text/Flickr_8k.testImages.txt'
# Read the validation image names in a set
# Read the test image names in a set
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))

# Create a list of all the test images with their full path names
test_img = []

for i in img: # img is list of full path names of all images
    if i[len(images):] in test_images: # Check if the image belongs to test set
        test_img.append(i) # Add it to the list of test images
print(len(test_img))

# load clean descriptions into memory
#add a startseq and endseq to the beginning and ending of each caption
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# descriptions
train_descriptions = load_clean_descriptions('image_captioning/descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

def preprocess(image_path):
    # Convert all the images to size 299x299 for the inception model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x

# Load the inception v3 model
model = InceptionV3(weights='imagenet')



# Create a new model, by removing the last layer (output layer) from the inception v3
model_new = Model(model.input, model.layers[-2].output)

# Function to encode a given image into a vector of size (2048, )
def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = model_new.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec

# Call the function to encode all the train images and store it as a pickle file, run it only once as it takes a lot of time

'''
start = time()
encoding_train = {}
for img in train_img:
    encoding_train[img[len(images):]] = encode(img)
print("Time taken in seconds =", time()-start)

# Save the bottleneck train features to disk
with open("image_captioning/encoded/encoded_train_images.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding_train, encoded_pickle)



# Call the function to encode all the test images 
start = time()
encoding_test = {}
for img in test_img:
    encoding_test[img[len(images):]] = encode(img)
print("Time taken in seconds =", time()-start)

with open("image_captioning/encoded/encoded_test_images.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding_test, encoded_pickle)
'''

train_features = load(open("image_captioning/encoded/encoded_train_images.pkl", "rb"))
print('Photos: train=%d' % len(train_features))

# Create a list of all the training captions
all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
len(all_train_captions)

#use only words that occur more than the 10 times in the vocabulary
word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

vocab_size = len(ixtoword) + 1 # one for appended 0's
print(vocab_size)

# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# data generator as dataset size is too large
def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    # loop forever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key+'.jpg']
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n==num_photos_per_batch:
                yield ([array(X1), array(X2)], array(y))
                X1, X2, y = list(), list(), list()
                n=0

# Load Glove vectors
glove_dir = 'image_captioning/glove/'
embeddings_index = {} # empty dictionary
f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


embedding_dim = 200

# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in wordtoix.items():
    #if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)

#create the model

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)



model.summary()

#set embedding layer weights and also as untrainable as the glove vectors are used there
print(model.layers[2])

model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False



model.compile(loss='categorical_crossentropy', optimizer='adam')

epochs = 10
number_pics_per_bath = 3
steps = len(train_descriptions)//number_pics_per_bath


#uncomment lines accordingly depending on whether a weights file is available
'''
for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, wordtoix, max_length, number_pics_per_bath)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
'''

#model.save('./model_weights/model_' + str(i) + '.h5')


new_model = tf.keras.models.load_model('./model_weights/model_try2_4.h5')
new_model.compile(loss='categorical_crossentropy', optimizer='adam')
'''
for i in range(20):
    generator = data_generator(train_descriptions, train_features, wordtoix, max_length, number_pics_per_bath)
    new_model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
new_model.save('./model_weights/model_try3_' + str(i) + '.h5')
'''

pred_model = tf.keras.models.load_model('./model_weights/model_try3_19.h5')

with open("image_captioning/encoded/encoded_test_images.pkl", "rb") as encoded_pickle:
    encoding_test = load(encoded_pickle)

#inference using greedy search algorithm
def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = pred_model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word

        if word == 'endseq':
            break

    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final



#inference using beam search algorithm
def beam_search_predictions(image, beam_index=3):
	start = [wordtoix["startseq"]]
	start_word = [[start, 0.0]]
	while len(start_word[0][0]) < max_length:
		temp = []
		for s in start_word:
			par_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
			preds = pred_model.predict([image, par_caps], verbose=0)
			word_preds = np.argsort(preds[0])[-beam_index:]
			# Getting the top <beam_index>(n) predictions and creating a
			# new list so as to put them via the model again
			for w in word_preds:
				next_cap, prob = s[0][:], s[1]
				next_cap.append(w)
				prob += preds[0][w]
				temp.append([next_cap, prob])

		start_word = temp
		# Sorting according to the probabilities
		start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
		# Getting the top words
		start_word = start_word[-beam_index:]

	start_word = start_word[-1][0]
	intermediate_caption = [ixtoword[i] for i in start_word]
	final_caption = []
	#break out of the loop if endseq has been reached
	for i in intermediate_caption:
		if i != 'endseq':
			final_caption.append(i)
		else:
			break

	final_caption = ' '.join(final_caption[1:])
	return final_caption


for z in range(10):
	pic = list(encoding_test.keys())[z]
	image = encoding_test[pic].reshape((1,2048))
	x=plt.imread(images+pic)
	plt.imshow(x)
	plt.show()
	print("Greedy:",greedySearch(image))
	print("Beam Search, K = 3:",beam_search_predictions(image, beam_index = 3))
	print("Beam Search, K = 5:",beam_search_predictions(image, beam_index = 5))
	print("Beam Search, K = 7:",beam_search_predictions(image, beam_index = 7))
	print("Beam Search, K = 10:",beam_search_predictions(image, beam_index = 10))




'''
pic = '2398605966_1d0c9e6a20.jpg'
image = encoding_test[pic].reshape((1,2048))
x=plt.imread('image_captioning/Images/'+pic)
plt.imshow(x)
plt.show()

print("Greedy Search:",greedySearch(image))
print("Beam Search, K = 3:",beam_search_predictions(image, beam_index = 3))
print("Beam Search, K = 5:",beam_search_predictions(image, beam_index = 5))
print("Beam Search, K = 7:",beam_search_predictions(image, beam_index = 7))
print("Beam Search, K = 10:",beam_search_predictions(image, beam_index = 10))
'''