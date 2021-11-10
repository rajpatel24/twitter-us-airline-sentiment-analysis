import re

import nltk
import pandas as pd

from nltk import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer

from nltk.corpus import sentiwordnet as swn

wordnet_lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', text)
    text = re.sub("(@[A-Za-z0-9_]+)", "", text)
    return text


def token_stop_pos(text):
    pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}

    tags = nltk.pos_tag(word_tokenize(text))

    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist


# lemmatization
def lemmatize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew


def sentiwordnet_analysis(pos_data):
    sentiment = 0
    tokens_count = 0
    for word, pos in pos_data:
        if not pos:
            continue

        lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
        if not lemma:
            continue

        synsets = wordnet.synsets(lemma, pos=pos)
        if not synsets:
            continue

        # Take the first sense, the most common
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        sentiment += swn_synset.pos_score() - swn_synset.neg_score()
        tokens_count += 1

        if not tokens_count:
            return 0
        if sentiment > 0:
            return "positive"
        if sentiment == 0:
            return "neutral"
        else:
            return "negative"


def sentiment_analysis():
    data = pd.read_csv("../Dataset/Tweets.csv")

    data = data[['tweet_id', 'airline_sentiment', 'text']]

    data['clean_text'] = data['text'].apply(clean_text)

    data['pos_tagged'] = data['clean_text'].apply(token_stop_pos)

    data['lemma'] = data['pos_tagged'].apply(lemmatize)

    data['sentiment_score'] = data['pos_tagged'].apply(sentiwordnet_analysis).values

    print("\n\n ------------------------ \n\n", data)

    data.to_csv("result.csv", columns=['tweet_id', 'text', 'airline_sentiment', 'sentiment_score'])


if __name__ == '__main__':
    sentiment_analysis()
