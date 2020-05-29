# Jai Ram G (जय राम जी) - Jiffy and intuitive Ramayan and Mahabharat Generator

An n-gram based model trained using deep learning techniques that can generate text in the style of a particular Indic Literature in Devnagari script

## Data Collection

* Ramayan sourced from https://github.com/svenkatreddy/Ramayana_Book
* Mahabharat scraped from https://www.sacred-texts.com/hin/mbs/ 

## Preprocessing

### Data Cleaning

* Cleaned for whitespaces, tabs, etc.
* Split into a set of characters for the model and creating the vocabulary for the corpus dataset
* Created one-hot vector representation for each character of the vocabulary

### Transliteration

* Transliteration from Devnagari script to English 
* Transliteration back to Devnagari Script once the model is trained

## Model

* Built a custom character level N-Gram natural language model using deep learning technique
* Took random n length strings from the corpus to predict the n+1$^{th}$ character using the LSTM model
