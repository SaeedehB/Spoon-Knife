# import nltk
# import csv
# from keras.datasets import mnist
# import  re, pprint
# import nltk
# import tokenizer


# import spacy
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
# from bs4 import BeautifulSoup
# from contractions import CONTRACTION_MAP
import unicodedata

# nlp = spacy.load('en_core', parse=True, tag=True, entity=True)
# nlp_vec = spacy.load('en_vecs', parse = True, tag=True, #entity=True)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

# nltk.download()
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import pickle


ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


remove_special_characters("Well this was fun! What do you think? 123#@!",
                          remove_digits=True)

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


remove_stopwords("The, and, if are stopwords, computer is not")

def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True,
                     text_lemmatization=True, special_char_removal=True,
                     stopword_removal=True, remove_digits=True):
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # lowercase the text
        if text_lower_case:
            doc = doc.lower()
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)

        # remove special characters and\or digits
        if special_char_removal:
            # insert spaces between special characters to isolate them
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)
            # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)

        normalized_corpus.append(doc)
    return normalized_corpus

corpus = normalize_corpus(ex, text_lower_case=False,
                          text_lemmatization=False, special_char_removal=False)
print(corpus)
# def preprocess(sent):
#     print('in def function')
#     sent = nltk.word_tokenize(sent)
#     sent = nltk.pos_tag(sent)
#     return sent
#
# sent = preprocess(ex)
# print('print sent')
# print(sent)
# grammar = "NP: {<JJ>}"
# cp = nltk.RegexpParser(grammar)
# result = cp.parse(sent)
print('-----------------print result---------------')
# print(result)



# mnist.load_data()

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# with open(r'C:\Users\bahrami\Desktop\work\Tadbir\TadbirAITask-master\Data\News.csv', mode='r') as csv_file:
#     sample = csv_file.read()
# sentences = nltk.sent_tokenize(sample,language='english')
# # tokens = nltk.word_tokenize(sentence)
# # tagged = nltk.pos_tag(tokens)