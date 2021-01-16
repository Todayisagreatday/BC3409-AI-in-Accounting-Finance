# Import libraries
import nltk
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from PIL import Image

# General cleaning
def general_cleaning(x):
    x = str(x)
    pattern = '[^a-zA-Z0-9\ ]'
    x = re.sub(pattern,'',x)
    x = x.lower()
    x = x.strip()
    return x

# Lemmatization
def get_lemmatized_text(corpus):
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

# Stopwords removal
stop = stopwords.words('english')
additional_stopwords = ["wa"]
stop = set(stop + additional_stopwords)
def remove_stop(x):
    x = word_tokenize(x)
    store = ''
    
    for i in x:
        if i not in stop:
            store += i + ' '
            
    return store

# Create and transform mask for wordcloud
def transform_format(val):
    if val == 0:
        return 255
    else:
        return val

rect_mask = np.array(Image.open("mask_2.png"))        
transformed_rect_mask = np.ndarray((rect_mask.shape[0],rect_mask.shape[1]), np.int32)
for i in range(len(rect_mask)):
    transformed_rect_mask[i] = list(map(transform_format, rect_mask[i]))

# Settings for positive wordcloud
def show_wordcloud_positive(data):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 500,
        max_font_size = 50, 
        scale = 3,
        random_state = 42,
        colormap = 'summer',
        mask=transformed_rect_mask
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()

# Settings for neutral wordcloud
def show_wordcloud_neutral(data):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 250,
        max_font_size = 50, 
        scale = 3,
        random_state = 42,
        colormap = 'Wistia',
        mask=transformed_rect_mask
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()

# Settings for negative wordcloud
def show_wordcloud_negative(data):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 250,
        max_font_size = 50, 
        scale = 3,
        random_state = 42,
        colormap = 'OrRd',
        mask=transformed_rect_mask
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()