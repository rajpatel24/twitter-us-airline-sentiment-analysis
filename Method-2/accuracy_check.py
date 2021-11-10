import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

df = pd.read_csv('result.csv')

df['sentiment_score'].replace('', np.nan, inplace=True)
df.dropna(subset=['sentiment_score'], inplace=True)

airline_sentiment = df.airline_sentiment
sentiments = df.sentiment_score

ascore = accuracy_score(airline_sentiment, sentiments)
print("\n\n Accuracy: ", ascore*100)
