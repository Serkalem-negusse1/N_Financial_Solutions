import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

class NLP_Analysis:
    def __init__(self, data):
        self.data = data
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(max_features=10)
    
    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return ' '.join(tokens)
    
    def add_processed_text(self):
        self.data['processed_headline'] = self.data['headline'].apply(self.preprocess_text)
        print("Processed text added!")
        print(self.data.head())
    
    def extract_keywords(self):
        X = self.vectorizer.fit_transform(self.data['processed_headline'])
        feature_names = self.vectorizer.get_feature_names_out()
        scores = X.sum(axis=0).A1
        keywords = dict(zip(feature_names, scores))
        return sorted(keywords.items(), key=lambda x: x[1], reverse=True)
    
    def perform_topic_modeling(self, num_topics=5):
        texts = self.data['processed_headline'].apply(lambda x: x.split())
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
        return lda_model.print_topics(num_words=5)
