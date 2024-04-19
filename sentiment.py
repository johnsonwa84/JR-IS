# import necessary libraries
import nltk
nltk.download('all')
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# original csv from R
df = pd.read_csv('sampleData.csv')


# send tweets to text in dataframe
text_items = df['Tweet'].tolist()


# nltk sentiment analyzer, will grade tweets based on positivity, etc.
sia = SentimentIntensityAnalyzer()

# initialize list for sentiment analysis scores
compound_scores = []

# Sentiment analysis scores, added to list (compound/combined as opposed to positive/negative)
for item in text_items:
    scores = sia.polarity_scores(item)
    compound_scores.append(scores['compound'])


# Create dataFrame from sentiment score list
sentiment_df = pd.DataFrame({'compound_score': compound_scores})


# merge scores with original stock dataset and save as csv for final dataset
merged_data = pd.concat([df, sentiment_df], axis=1)
print(merged_data)
merged_data.to_csv('cleanData.csv')