# pip install streamlit nltk spacy pandas newspaper3k gensim pyLDAvis matplotlib seaborn plotly

import streamlit as st
import nltk
import spacy
import pandas as pd
import numpy as np
from newspaper import Article
import re
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import plotly.graph_objects as go
import plotly.express as px

# Function to check and download NLTK data
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

# Function to check and download spaCy model
def download_spacy_model():
    if not spacy.util.is_package('en_core_web_sm'):
        spacy.cli.download('en_core_web_sm', quiet=True)

# Download necessary resources if not already present
download_nltk_data()
download_spacy_model()

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

nlp = spacy.load('en_core_web_sm')

def scrape_article(url):
    """Scrapes text content from a given URL."""
    article = Article(url)
    article.download()
    article.parse()
    return article.text, article.title

def preprocess_text(text):
    """Preprocesses text by tokenizing, removing stopwords, and lemmatizing."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and token.text not in stop_words]
    return tokens

def apply_lda(texts, num_topics=5):
    """Applies LDA to the preprocessed text corpus."""
    dictionary = corpora.Dictionary([texts])
    corpus = [dictionary.doc2bow(texts)]

    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    return lda_model, corpus, dictionary

def visualize_topics(lda_model, corpus, dictionary):
    """Visualizes the topics using pyLDAvis and returns the HTML string."""
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    return pyLDAvis.prepared_data_to_html(vis_data)

def plot_topic_word_distribution(lda_model):
    """Creates an interactive heatmap of word distribution across topics."""
    topic_words = []
    for idx, topic in lda_model.print_topics(-1):
        topic_words.append([w for w, _ in lda_model.show_topic(idx, topn=10)])

    df = pd.DataFrame(topic_words).T
    df.columns = [f'Topic {i+1}' for i in range(len(topic_words))]
    
    fig = px.imshow(df.notna(), 
                    labels=dict(x="Topics", y="Top Words", color="Word Presence"),
                    x=df.columns,
                    y=df.index,
                    color_continuous_scale='YlOrRd')
    
    fig.update_layout(title='Distribution of Top Words Across Topics',
                      xaxis_title='Topics',
                      yaxis_title='Top Words')
    
    return fig

def plot_topic_prevalence(lda_model, corpus):
    """Creates a bar chart of topic prevalence."""
    topic_prevalence = [0] * lda_model.num_topics
    for doc in corpus:
        topic_dist = lda_model.get_document_topics(doc)
        for topic, prob in topic_dist:
            topic_prevalence[topic] += prob

    topic_prevalence = [p / sum(topic_prevalence) for p in topic_prevalence]
    
    fig = go.Figure(data=[go.Bar(x=[f'Topic {i+1}' for i in range(lda_model.num_topics)],
                                 y=topic_prevalence,
                                 text=[f'{p:.2%}' for p in topic_prevalence],
                                 textposition='outside')])
    fig.update_layout(title='Topic Prevalence',
                      xaxis_title='Topics',
                      yaxis_title='Prevalence',
                      yaxis_tickformat='.2%')
    return fig

def main():
    st.set_page_config(layout="wide")
    st.title("Comprehensive LDA Topic Modeling and Visualization App")

    st.sidebar.header("Input")
    url = st.sidebar.text_input("Enter the URL of a historical news article or event:")
    num_topics = st.sidebar.slider("Number of topics", min_value=2, max_value=10, value=5)

    if st.sidebar.button("Analyze"):
        with st.spinner("Analyzing the article..."):
            # Scrape and preprocess the article
            article_text, article_title = scrape_article(url)
            preprocessed_text = preprocess_text(article_text)

            # Apply LDA
            lda_model, corpus, dictionary = apply_lda(preprocessed_text, num_topics=num_topics)

            # Main content area
            st.header(f"Analysis Results for: {article_title}")
            
            # Topics Extracted
            st.subheader("1. Topics Extracted")
            st.write("The LDA model has identified the following main topics in the article:")
            topics_df = pd.DataFrame(columns=["Top Words"])
            for idx, topic in lda_model.print_topics(-1):
                topic_words = ", ".join([word.split("*")[1].strip().strip('"') for word in topic.split("+")])
                topics_df.loc[f"Topic {idx+1}"] = topic_words
            st.table(topics_df)
            
            # Topic Prevalence
            st.subheader("2. Topic Prevalence")
            st.write("This chart shows how dominant each topic is in the overall article:")
            prevalence_fig = plot_topic_prevalence(lda_model, corpus)
            st.plotly_chart(prevalence_fig, use_container_width=True)
            
            # Word Distribution
            st.subheader("3. Distribution of Top Words Across Topics")
            st.write("This heatmap visualizes which words are important to which topics:")
            heatmap_fig = plot_topic_word_distribution(lda_model)
            st.plotly_chart(heatmap_fig, use_container_width=True)

            # Interactive Topic Visualization
            st.subheader("4. Interactive Topic Visualization")
            st.write("Explore the topics and their relationships in more detail:")
            vis_html = visualize_topics(lda_model, corpus, dictionary)
            st.components.v1.html(vis_html, width=1300, height=800)

            # Interpretation Guide
            with st.expander("How to Interpret These Results"):
                st.write("""
                1. **Topics Extracted**: Each topic is represented by a set of words. Words with higher weights (not shown) contribute more to the topic.
                2. **Topic Prevalence**: This shows how much each topic contributes to the overall document. Higher bars indicate more prevalent topics.
                3. **Word Distribution**: The heatmap shows which words are important to which topics. Darker cells indicate stronger associations.
                4. **Interactive Visualization**: This plot allows you to explore the relationships between topics and terms in more detail.
                   - The left panel shows the topics as circles. Size indicates prevalence.
                   - The right panel shows the top terms for the selected topic.
                   - You can adjust the λ value to change the relevance metric for terms.
                """)

if __name__ == '__main__':
    main()