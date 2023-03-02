## Essentials
import re
import pandas as pd
import string
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4');

#Emoji handling
import emoji

# Defining dictionary containing all emojis with their meanings.
emoji_dict = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

## Defining set containing all stopwords in english.
stop_words = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

def preprocess(tweet):
    # Create Lemmatizer and Stemmer.
    # nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern        = r"[^ ]+\.[^ ]+"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    emoji_pattern = r":([^:]+):(.*?)"
    replace_emoji_pattern = "EMOJI\\1 "
    
    tweet = tweet.lower()
        
    # Replace all URls with 'URL'
    tweet = re.sub(urlPattern,' URL',tweet)
    
    # Replace all emojis.
    for emo in emoji_dict.keys():
        tweet = tweet.replace(emo, "EMOJI" + emoji_dict[emo])    
    
    # Replace @USERNAME to 'USER'.
    tweet = re.sub(userPattern,' USER', tweet) 
    
    # Remove punctuations
    exclude = string.punctuation
    tweet = tweet.translate(str.maketrans('', '', exclude))
    
    # Replace all emojis in unicode form.
    tweet = emoji.demojize(tweet)
    tweet = re.sub(emoji_pattern, replace_emoji_pattern, tweet)
    
    # Replace all non alphabets.
    tweet = re.sub(alphaPattern, " ", tweet)
    
    # Replace 3 or more consecutive letters by 2 letter.
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

    tweetwords = ''
    
    for word in tweet.split():
        # Checking if the word is a stopword.
        if word not in stop_words:
            if len(word) > 1:
                # Lemmatizing the word.
                word = lemmatizer.lemmatize(word)
                tweetwords += (word+' ')
    return tweetwords

def load_models():
    '''
    Replace '..path/' by the path of the saved models.
    '''
    
    # Load the vectoriser.
    file = open('./models/vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    # Load the LR Model.
    file = open('./models/Sentiment-LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()
    
    return vectoriser, LRmodel

def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform(list(map(preprocess, text)))
    sentiment = model.predict(textdata)
    
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
        
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df

def sentiment_table(selected_user, df):
    if selected_user != 'Overall':
        df = df[df["Author"] == selected_user]

    vectorizer, model = load_models()

    sent_df = predict(vectorizer, model, df.Message)
    x = sent_df.sentiment.value_counts()
    return x


