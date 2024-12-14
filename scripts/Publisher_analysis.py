import pandas as pd
from collections import Counter
from urllib.parse import urlparse

class PublisherAnalysis:
    def __init__(self, data):
        self.data = data
        
    def count_articles_per_publisher(self):
        """Count the number of articles each publisher has published."""
        return self.data['publisher'].value_counts()

    def extract_domains(self):
        """Extract the domains from email addresses or URLs in the publisher field."""
        self.data['domain'] = self.data['publisher'].apply(self._extract_domain)
    
    def _extract_domain(self, publisher):
        """Helper function to extract domain from publisher."""
        if '@' in publisher:
            return publisher.split('@')[-1]
        elif publisher.startswith('http'):
            return urlparse(publisher).netloc
        return publisher
    
    def count_articles_per_domain(self):
        """Count the number of articles per unique domain."""
        return self.data['domain'].value_counts()

