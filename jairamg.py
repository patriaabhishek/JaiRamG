
# !pip install indic_transliteration

import pandas as pd
import json

import urllib.request
import os
import zipfile
import glob
import numpy as np
import random
import string

# %tensorflow_version 1.15
import tensorflow as tf
from keras import models
from keras import layers
from keras import callbacks
from keras.utils import plot_model
import matplotlib.pyplot as plt
import progressbar
from indic_transliteration import sanscript

#HYPERPARAMETERS

TRANSLITERATE = False
INPUT_LENGTH = 40
OUTPUT_LENGTH = 1
DATASET_SIZE = 100000
NUM_EPOCHS = 100
BATCH_SIZE = 512
HIDDEN_SIZE = 350
GEN_LENGTH = 3000

def get_mahabharat_text():
  text = ""
  nums = list(range(1, 18))
  for num in nums:
    file_name = "book{:02d}.txt".format(11)
    text_file = open(file_name, "r")
    lines = text_file.readlines()
    data = " ".join(lines)
    text = text + data

  return text

def get_ramayan_data():
  kands = ['balakanda', 'ayodhyakanda', 'aranyakanda', 'kishkindhakanda', 'sundarakanda', 'uttarakanda']
  data_list = []
  for kand in kands:
    file_name = kand + '_all.json' 
    with open(file_name) as f:
      data = json.load(f)
      data_list.append(data[kand]['chapters'])
      # print(data)  
  return data_list

def get_ramayan_text():
  data_text = ""
  data_list = get_ramayan_data()
  for kand in data_list:
    for chapter in kand:
      slokas = chapter['slokas']
      for slok in slokas:
        data_text = data_text + slok

  return data_text

def download_hindi_literature():
  corpus_path = "corpus"
  corpus_url = "https://github.com/cltk/hindi_text_ltrc/archive/master.zip"
  corpus_zip_path = "master.zip"
  urllib.request.urlretrieve(corpus_url, corpus_zip_path)

  # Unzip the whole git-repository to the corpus-path.
  print("\n\nUnzipping corpus...\n\n")
  zip_file = zipfile.ZipFile(corpus_zip_path, 'r')
  zip_file.extractall(corpus_path)
  zip_file.close()

def get_literature_text():
  glob_path = os.path.join(corpus_path, "**/*.txt")
  paths = glob.glob(glob_path, recursive=True)
  
  text = ""
  for file_name in paths:
    text_file = open(file_name, "r")
    lines = text_file.readlines()
    data = " ".join(lines)
    text = text + data

  return text

def create_vocab(text):
  return sorted(list(set(text)))

def clean_text(text, alpha = True):
    if alpha:    
      letters = string.ascii_letters
      for letter in letters:
        text = text.replace(letter, " ")  

    text = text.replace("\t", " ")
    text = text.replace("\n", " ")
    text = text.replace("।", " ")
    text = text.replace("0", " ")
    text = text.replace("1", " ")
    text = text.replace("2", " ")
    text = text.replace("3", " ")
    text = text.replace("4", " ")
    text = text.replace("5", " ")
    text = text.replace("6", " ")
    text = text.replace("7", " ")
    text = text.replace("8", " ")
    text = text.replace("9", " ")
    text = " ".join(text.split())
    
    return text

def random_substring_of_length(text, length):
    start_index = random.randint(0, len(text) - length)
    return text[start_index: start_index + length]

def transliterate(text, toDevanagari = False):
  if toDevanagari:
    text = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
  else:
    text = sanscript.transliterate(text, sanscript.IAST, sanscript.DEVANAGARI)  
  return text

def encode_string(text, vocab_set):
    encoded_string = []
    for character in text:
        encoded_character = np.zeros((len(vocab_set),))
        one_hot_index = vocab_set.index(character)
        encoded_character[one_hot_index] = 1
        encoded_string.append(encoded_character)
    return np.array(encoded_string)

def get_index_from_prediction(prediction, temperature=0.0):    
    if temperature == 0.0:
        return np.argmax(prediction)
    else:
        prediction = np.asarray(prediction).astype('float64')
        prediction = np.log(prediction) / temperature
        exp_prediction= np.exp(prediction)
        prediction = exp_prediction / np.sum(exp_prediction)
        probabilities = np.random.multinomial(1, prediction, 1)
        return np.argmax(probabilities)

def decode_sequence(sequence, vocab_set, temperature=0.0):
    result_string = ""
    for element in sequence:
        index = get_index_from_prediction(element)
        character = vocab_set[index]
        result_string += character
    return result_string

def generate_output(model, vocab_set, text):   
    random_string = random_substring_of_length(text, INPUT_LENGTH)
    result_string = random_string
    #print("Seed string:  ", random_string)
    input_sequence = encode_string(random_string, vocab_set)
 
    # Generate a string.
    while len(result_string) < GEN_LENGTH:
        output_sequence = model.predict(np.expand_dims(input_sequence, axis=0))
        output_sequence = output_sequence[0]
        decoded_string = decode_sequence(output_sequence, vocab_set)
        output_sequence = encode_string(decoded_string, vocab_set)
        result_string += decoded_string
        input_sequence = input_sequence[OUTPUT_LENGTH:]
        input_sequence = np.concatenate((input_sequence, output_sequence), axis=0)

    return result_string

def create_data(text, vocab_set):
    data_input = []
    data_output = []
    current_size = 0
    bar = progressbar.ProgressBar(max_value=DATASET_SIZE)
    print("\n\nGenerating data set...\n\n")
    while current_size < DATASET_SIZE:        
        random_string = random_substring_of_length(text, INPUT_LENGTH + OUTPUT_LENGTH)
        random_string_encoded = encode_string(random_string, vocab_set)
 
        input_sequence = random_string_encoded[:INPUT_LENGTH]
        output_sequence = random_string_encoded[INPUT_LENGTH:]
 
        data_input.append(input_sequence)
        data_output.append(output_sequence)
 
        current_size += 1
        bar.update(current_size)
    bar.finish()
     
    train_input = np.array(data_input)
    train_output = np.array(data_output)
    return train_input, train_output

def plot_history(history):
    """ Plots the history. """
 
    # Render the accuracy.
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("accuracy.png")
    plt.clf()
 
    # Render the loss.
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss.png")

def create_model(vocab_set):   
    input_shape = (INPUT_LENGTH, len(vocab_set))
 
    model = models.Sequential()
    model.add(layers.LSTM(HIDDEN_SIZE, input_shape=input_shape, activation="relu"))
    model.add(layers.Dense(OUTPUT_LENGTH * len(vocab_set), activation="relu"))
    model.add(layers.Reshape((OUTPUT_LENGTH, len(vocab_set))))
    model.add(layers.TimeDistributed(layers.Dense(len(vocab_set), activation="softmax")))
    model.summary()
 
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
 
    return model

def main_fn(text, transliterate = TRANSLITERATE):
  
  if TRANSLITERATE:
    text = transliterate(text)
    text = clean_text(text, alpha = False)
  else:  
    text = clean_text(text, alpha = True)
  
  vocab_set = create_vocab(text)
  train_input, train_output = create_data(text, vocab_set)
  model = create_model(vocab_set)
  
  fit = model.fit(train_input, 
                  train_output,
                  epochs = NUM_EPOCHS, 
                  batch_size = BATCH_SIZE,        
        )
  out = generate_output(model, vocab_set, text)
  return out, fit

ramanayan_text = get_ramayan_text()
out, acc = main_fn(ramanayan_text)
print(out)
plot_history(acc)

mahabharat_text = get_mahabharat_data()
out, acc = main_fn(mahabharat_text)
print(out)
plot_history(acc)

lit_text = get_literature_text()
out, acc = main_fn(lit_text)
print(out)
plot_history(acc)