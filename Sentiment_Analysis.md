# Sentiment Analysis
Code:  https://github.com/udacity/deep-learning-v2-pytorch/sentiment-analysis-network.

## 1. Develop a Predictive Theory
test your theory of what features of a review correlate with the label

- common words like "the" appear very often in both positive and negative reviews. Instead of finding the most common words in positive or negative reviews, what you really want are the words found in positive reviews more often than in negative reviews, and vice versa. To accomplish this, you'll need to calculate the ratios of word usage between positive and negative reviews.
