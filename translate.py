'''Translate text using google translate and then 
    do sentiment analysis on translated text.

Usage:
    python translate.py config_translate.yaml

Author:
    Cedric Basuel
'''

import sys
import yaml
import googletrans
from googletrans import Translator
from google_trans_new import google_translator
import nltk
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

import random
import pandas as pd
from sentiment_analysis_via_api import get_sentiment_score_using_google, get_microsoft_score
from collections import Counter

nltk.download('vader_lexicon')


def translate_text_list(text_list):
    translator = Translator()
    text_translated = []
    translations = translator.translate(text_list)

    for translation in translations:
        text_translated.append(translation.text)

    return text_list, text_translated


def translate_text_list_1(text_list):
    '''googletrans appears to have a bug right now.'''

    translator = google_translator()

    translations = [translator.translate(text) for text in text_list]
        
    return text_list, translations


def get_words_not_translated(text_translated):

    untranslated_words = []

    for text in text_translated:
        for word in text:
            if word not in english_words:
                untranslated_words.append(word)
    
    return untranslated_words


if __name__ == '__main__':

    CONFIG_FILE = sys.argv[1]

    with open(CONFIG_FILE) as cfg:
        config = yaml.safe_load(cfg)

    # get tweets with low english word ratio
    df = pd.read_csv(config['tweets'])
    df_low_english = df[df['english_word_count'] < 0.20]
    print('len df_low_eng', df_low_english.shape)

    # generate random index
    random.seed(10)
    num_sample = config['num_sample']
    indices = random.sample(range(len(df_low_english)), num_sample)
    df_to_translate = df_low_english.iloc[indices,:]

    # translate to english
    text_to_translate = list(df_to_translate['text'].copy())
    _, text_translated = translate_text_list_1(text_to_translate)

    # get sentiments from Google NL, VADER, MS Text Analytics
    # Google NL
    text_translated_, text_sentiment, text_google_scores = get_sentiment_score_using_google(text_translated)

    # VADER
    analyzer=SentimentIntensityAnalyzer()
    text_vader_scores = []

    for text in text_translated:
        temp_score = analyzer.polarity_scores(text)
        text_vader_scores.append(temp_score['compound'])
    
    # MS Text Analytics
    credential=AzureKeyCredential(config['azure']['credential'])
    text_analytics_client=TextAnalyticsClient(endpoint=config['azure']['client'], credential=credential)

    _, ms_pos, ms_neg, ms_neutral = get_microsoft_score(text_translated)

    # Append scores to df_to_translate
    df_to_translate['text_translated'] = text_translated
    df_to_translate['new_vader_score'] = text_vader_scores
    df_to_translate['new_google_score'] = text_google_scores
    df_to_translate['new_microsoft_score_pos'] = ms_pos
    df_to_translate['new_microsoft_score_neg'] = ms_neg
    df_to_translate['new_microsoft_score_neu'] = ms_neutral

    df_to_translate.to_csv(config['output'])









