import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Download necessary nltk resources
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Function to scrape reviews
def scrape_reviews(url, model_name):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"❌ Failed to fetch reviews for {model_name} from {url}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    reviews = []

    # Try extracting review text from different HTML elements
    for review in soup.find_all('span', {'data-hook': 'review-body'}):
        text = review.text.strip()
        reviews.append({'Model': model_name, 'Review': text})

    if not reviews:  # Alternative method in case the above fails
        for div in soup.find_all('div', class_='review-text-content'):
            text = div.text.strip()
            reviews.append({'Model': model_name, 'Review': text})

    # Debugging print
    print(f"✅ Scraped {len(reviews)} reviews for {model_name}")

    return reviews

# iPhone models and their Amazon review URLs
iphone_models = {
    'iPhone 14': 'https://www.amazon.in/Apple-iPhone-14-128GB-Blue/dp/B0BDK62PDX',
    'iPhone 14 Pro': 'https://www.amazon.com/Apple-iPhone-14-Pro-128GB/dp/B0BN95FRW9',
    'iPhone 14 Pro Max': 'https://www.amazon.com/Apple-iPhone-14-Pro-Max/dp/B0BN93P98N',
    'iPhone 15': 'https://www.amazon.in/Apple-iPhone-15-128-GB/dp/B0CHX1W1XY?th=1',
    'iPhone 15 Pro': 'https://www.amazon.in/Apple-iPhone-15-Pro-128/dp/B0CHX2DRGV?th=1',
    'iPhone 15 Pro Max': 'https://www.amazon.in/Apple-iPhone-Pro-Max-256/dp/B0CHWV2WYK',
    'iPhone 16': 'https://www.amazon.in/iPhone-16-128-GB-Control/dp/B0DGJHBX5Y?th=1',
    'iPhone 16 Pro': 'https://www.amazon.in/iPhone-16-Pro-128-GB/dp/B0DGJ7X1DX?th=1',
    'iPhone 16 Pro Max': 'https://www.amazon.in/iPhone-16-Pro-Max-256/dp/B0DGHYDZR9?th=1'
}

# Scrape reviews for all models
iphone_reviews = []
for model, url in iphone_models.items():
    iphone_reviews += scrape_reviews(url, model)

# Convert to DataFrame
df = pd.DataFrame(iphone_reviews)

# Check if DataFrame is empty
if df.empty:
    print("⚠️ No reviews scraped. Check URLs or parsing logic.")
else:
    print(df.head())  # Show first few rows

# Sentiment Analysis
def analyze_sentiment(text):
    return sia.polarity_scores(text)['compound']

if not df.empty:
    df['Sentiment Score'] = df['Review'].apply(analyze_sentiment)
    df['Sentiment Label'] = df['Sentiment Score'].apply(lambda x: 'Positive' if x > 0.05 else 'Negative' if x < -0.05 else 'Neutral')

    # Visualization: Sentiment Distribution per Model
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x='Model', hue='Sentiment Label', palette='coolwarm')
    plt.title('Sentiment Analysis of iPhone Models')
    plt.xticks(rotation=45)
    plt.show()

    # Word Cloud for common words
    def generate_wordcloud(sentiment):
        text = ' '.join(df[df['Sentiment Label'] == sentiment]['Review'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{sentiment} Sentiment WordCloud')
        plt.show()

    generate_wordcloud('Positive')
    generate_wordcloud('Negative')

    # Save to CSV for reference
    df.to_csv('iphone_reviews_sentiment.csv', index=False)
    print("✅ Sentiment analysis saved to 'iphone_reviews_sentiment.csv'")