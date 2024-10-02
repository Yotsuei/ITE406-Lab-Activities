import sys
import nltk
import ssl
import re
from bs4 import BeautifulSoup
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import Flask, render_template, request

# Set up Flask
app = Flask(__name__)

# Function to download NLTK data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("NLTK data not found. Attempting to download...")
        try:
            # Try to create an unverified HTTPS context
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            nltk.download('punkt')
            nltk.download('stopwords')
            print("NLTK data downloaded successfully.")
        except Exception as e:
            print(f"Failed to download NLTK data: {e}")
            print("Please download the required NLTK data manually.")
            sys.exit(1)

# Download NLTK data
download_nltk_data()

# Function to scrape article text from a URL
def get_article_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    article_text = ""
    for p in soup.find_all('p'):
        article_text += p.get_text() + " "
    return article_text

# Preprocess the text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w in stop_words]
    return " ".join(filtered_tokens)

# Analyze article: word frequency and sentiment analysis
def analyze_article(text):
    tokens = word_tokenize(text)
    word_counts = Counter(tokens)
    top_words = word_counts.most_common(10)

    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)

    return {
        "top_words": top_words,
        "sentiment": sentiment_scores
    }

# Summarize the article
def summarize_text(article_text):
    sentences = sent_tokenize(article_text)
    processed_text = preprocess_text(article_text)
    word_tokens = word_tokenize(processed_text)
    word_freq = Counter(word_tokens)
    top_words = [word for word, freq in word_freq.most_common(10)]
    summary_sentences = [sentence for sentence in sentences if any(word in sentence.lower() for word in top_words)]
    summary = " ".join(summary_sentences)
    return summary

# Flask route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Flask route to handle article analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    url = request.form['url']
    try:
        # Get article, preprocess, and analyze
        article_text = get_article_text(url)
        # article_text = "Kansas City's Erik Thommy equalised to take the game to extra time, when Mexican defender Omar Campos and Sierra Leonean forward Kei Kamara scored to seal the win. For Giroud, who retired from France duty in July as the nation's record scorer with 57 goals, it was important after August's Leagues Cup final defeat by Columbus Crew."
        processed_text = preprocess_text(article_text)
        analysis = analyze_article(processed_text)
        summary = summarize_text(article_text)

        # Total word counts
        total_words_full = len(word_tokenize(article_text))
        total_words_summary = len(word_tokenize(summary))

        return render_template('result.html', 
                               article_text=article_text,
                               summary=summary,
                               top_words=analysis['top_words'],
                               sentiment=analysis['sentiment'],
                               total_words_full=total_words_full,
                               total_words_summary=total_words_summary)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)  # This will print to the console for debugging
        return render_template('error.html', error=error_message)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)