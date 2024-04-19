# JR-IS

Neural Network Prediction of TSLA Stock Price based on Twitter Sentiment

This project utilizes a dataset from Kaggle consisting of Tweets from 2021/2022 containing keywords relating to "Tesla". The R code file 'dataset' cleans this original Kaggle dataset to include a sample of tweets relating to Tesla, and also merges these tweets with the stock price of Tesla at that given timestamp. The 'sentiment' python file imports the cleaned dataset from R and performs sentiment analysis on each of the tweets, before then appending the sentiment scores to the dataset, creating a final dataset to be the input for the neural network. The 'model' python file imports the final dataset from 'sentiment' and uses Tensorflow to build a neural network that takes stock price and sentiment score as input features, and attempts to predict future stock price (2 weeks into the future) as the target output.


