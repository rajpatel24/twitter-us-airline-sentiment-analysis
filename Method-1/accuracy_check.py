import pandas as pd
from sklearn.metrics import accuracy_score

df = pd.read_csv('result.csv')

airline_sentiment = df.airline_sentiment
sentiments = df.sentiment_result

ascore = accuracy_score(airline_sentiment, sentiments)
print("\n\n Accuracy: ", ascore*100)
