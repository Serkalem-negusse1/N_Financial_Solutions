import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

class EDA_Analysis:
    def __init__(self, data):
        self.data = data
        self.sia = SentimentIntensityAnalyzer()

    def descriptive_statistics(self):
        # Calculate lengths of each headline
        self.data['headline_length'] = self.data['headline'].apply(len)
        
        # Calculate basic statistics for headline length
        mean_length = np.mean(self.data['headline_length'])
        median_length = np.median(self.data['headline_length'])
        max_length = np.max(self.data['headline_length'])
        min_length = np.min(self.data['headline_length'])

        # Return the statistics as a dictionary
        return {
            'mean_length': mean_length,
            'median_length': median_length,
            'max_length': max_length,
            'min_length': min_length
        }
    
    def articles_per_publisher(self):
        # Count the number of articles per publisher
        articles_count = self.data['publisher'].value_counts()
        
        # Return as a dictionary
        return articles_count.to_dict()
    
    def publication_trends(self):
        # Convert 'date' column to datetime format
        # # Use the format parameter if you know the format of your date strings
        self.data['date'] = pd.to_datetime(self.data['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        # Analyze publication dates to see trends over time
        
        # Example: count articles per day
        articles_per_day = self.data['date'].dt.date.value_counts().sort_index()

        # Return the time series data as a dictionary
        return articles_per_day.to_dict()
    
    def perform_sentiment_analysis(self):
        # Calculate sentiment scores
        self.data['sentiment'] = self.data['headline'].apply(lambda x: self.sia.polarity_scores(x)['compound'])
        
        # Categorize sentiment scores
        self.data['sentiment_category'] = pd.cut(
            self.data['sentiment'],
            bins=[-1, -0.5, -0.0001, 0.5, 1],
            labels=['Very Negative', 'Negative', 'Neutral', 'Positive']
        )

    def get_sentiment_distribution(self):
        return self.data['sentiment_category'].value_counts()

    def filter_stock_data(self, stock_symbol):
        return self.data[self.data['stock'] == stock_symbol]
