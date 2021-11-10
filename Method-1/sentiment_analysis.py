import re
import string
import nltk
import pandas as pd

from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet, stopwords

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

negative_words = {"needn't", "hasn't", "no", "mightn't", "mustn't", "needn", "wasn't", "shouldn't", "couldn't", "don't",
                  "didn't", "weren't", "shouldn", "nor", "wouldn't", "didn", "hadn", "don", "isn", "won", "isn't",
                  "mightn", "weren", "haven", "couldn", "hadn't", "wasn", "mustn", "aren't", "hasn", "doesn't", "won't",
                  "doesn", "aren", "not", "wouldn", "shan't", "haven't", "ain", "shan"}


def sentiment_analysis():
    df = pd.read_csv("../Dataset/Tweets.csv")

    df = df[['tweet_id', 'airline_sentiment', 'text']]
    df['sentiment_result'] = 'none'

    stop_words = set(stopwords.words('english')) - negative_words

    for i in range(len(df)):
        # extracting the paragraph
        sentence = df.text[i]

        # removing hyperlinks
        sentence = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                      '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', sentence)

        # removing twitter usernames which are preceded by a '@'
        sentence = re.sub("(@[A-Za-z0-9_]+)", "", sentence)

        # remove punctuation
        sentence = sentence.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
        words = re.split(r"[\s+.]", sentence)
        tokens = filter(None, words)
        tokens = [word for word in tokens]

        words = [word for word in tokens if not word.lower() in stop_words]

        pos_tagged = nltk.pos_tag(words)

        newlist = []
        pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}
        for word, tag in pos_tagged:
            newlist.append(tuple([word, pos_dict.get(tag[0])]))

        sentiment = 0
        for word, tag in newlist:
            if not tag:
                continue

            lemma = wordnet_lemmatizer.lemmatize(word, pos=tag)
            if not lemma:
                continue

            synsets = wordnet.synsets(lemma, pos=tag)
            if not synsets:
                continue

            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            sentiment += swn_synset.pos_score() - swn_synset.neg_score()

            if sentiment > 0.05:
                df['sentiment_result'][i] = 'positive'
            elif sentiment < -0.05:
                df['sentiment_result'][i] = 'negative'
            else:
                df['sentiment_result'][i] = 'neutral'

    df.to_csv('result.csv', columns=['tweet_id', 'text', 'airline_sentiment', 'sentiment_result'])


if __name__ == '__main__':
    sentiment_analysis()
