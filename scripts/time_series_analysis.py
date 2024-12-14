import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

class TimeSeriesAnalysis:
    def __init__(self, data):
        self.data = data 
        # Convert 'date' to datetime, assuming ISO8601 format
        self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce', utc=True)
        self.data.set_index('date', inplace=True)
        
    def aggregate_data(self, freq='D'):
        """Aggregate data to the specified frequency."""
        self.aggregated_data = self.data.resample(freq).size()

    def perform_decomposition(self, period):
        """Perform seasonal decomposition on the aggregated data."""
        self.decomposition = seasonal_decompose(self.aggregated_data, model='additive', period=period)

    def plot_decomposition(self):
        """Plot the original data and decomposition components."""
        plt.figure(figsize=(14, 10))

        plt.subplot(411)
        plt.plot(self.aggregated_data, label='Original', color='green')
        plt.legend(loc='upper left')
        plt.title('Daily Publication Frequency')

        plt.subplot(412)
        plt.plot(self.decomposition.trend, label='Trend', color='yellow')
        plt.legend(loc='upper left')
        plt.title('Trend Component')

        plt.subplot(413)
        plt.plot(self.decomposition.seasonal, label='Seasonal', color='blue')
        plt.legend(loc='upper left')
        plt.title('Seasonal Component')

        plt.subplot(414)
        plt.plot(self.decomposition.resid, label='Residual/Irregular', color='red')
        plt.legend(loc='upper left')
        plt.title('Residual Component')

        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Update with your file path
    file_path = 'E:/10Academy/Data/raw_analyst_ratings/raw_analyst_ratings.csv'
    
    # Load the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path)
    
    # Pass the DataFrame to the TimeSeriesAnalysis class
    ts_analysis = TimeSeriesAnalysis(data)
    ts_analysis.aggregate_data(freq='D')  # Daily frequency
    ts_analysis.perform_decomposition(period=365)  # Assuming yearly seasonality
    ts_analysis.plot_decomposition()
