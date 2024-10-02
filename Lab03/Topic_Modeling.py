import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Sample corpus for demonstration (short news articles)
corpus = [
    """The stock market rallied today as tech giants reported strong earnings. 
    Apple, Google, and Microsoft all beat expectations, driving the Nasdaq to new highs. 
    Investors are optimistic about the continued growth in the tech sector, despite ongoing 
    concerns about inflation and supply chain disruptions.""",

    """Scientists have made a breakthrough in renewable energy with a new type of solar panel. 
    The panel, developed by researchers at MIT, is 50% more efficient than current models. 
    This innovation could significantly reduce the cost of solar energy and accelerate the 
    transition to sustainable power sources.""",

    """A major winter storm is expected to hit the East Coast this weekend. 
    Meteorologists predict heavy snowfall and strong winds, potentially causing 
    power outages and travel disruptions. Residents are advised to stock up on 
    essentials and avoid unnecessary travel during the storm.""",

    """The World Health Organization has approved a new vaccine for malaria. 
    This breakthrough could save tens of thousands of lives annually, particularly 
    in Africa where the disease is most prevalent. The vaccine has shown 75% efficacy 
    in preventing severe malaria in young children.""",

    """In sports news, the underdog team has won the championship in a stunning upset. 
    The victory came after a nail-biting final match that went into overtime. 
    Fans around the world celebrated the team's first title in 50 years, marking 
    a historic moment in the sport."""
]

def create_topic_model(corpus, num_topics=2, num_words=5):
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(corpus)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)
    
    words = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append(top_words)
    return topics

class TopicModelingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Topic Modeling Presentation")
        self.master.geometry("1000x700")
        
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=True, fill="both")
        
        self.create_introduction_tab()
        self.create_key_concepts_tab()
        self.create_lda_tab()
        self.create_demo_tab()
        self.create_quiz_tab()
    
    def create_tab(self, title):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=title)
        return tab
    
    def create_scrolled_text(self, parent, content):
        text_area = scrolledtext.ScrolledText(parent, wrap=tk.WORD, width=110, height=35)  # Increased size
        text_area.insert(tk.END, content)
        text_area.config(state=tk.DISABLED)
        text_area.pack(padx=10, pady=10)
    
    def create_introduction_tab(self):
        tab = self.create_tab("Introduction")
        content = """Topic Modeling: Uncovering Hidden Themes in Text

Topic Modeling is an unsupervised machine learning technique used to discover hidden themes or topics in a collection of documents. It's a powerful tool for analyzing large volumes of text data and extracting meaningful insights.

Key applications of Topic Modeling include:
1. Content recommendation systems
2. Document clustering and organization
3. Trend analysis in social media and news
4. Information retrieval and search engine optimization
5. Text summarization

Topic modeling algorithms are designed to uncover the underlying semantic structure in a document collection, making it easier to organize, search, and understand large amounts of textual information.

The power of topic modeling lies in its ability to automatically identify themes that might not be immediately apparent to human readers, especially when dealing with large volumes of text."""
        self.create_scrolled_text(tab, content)
    
    def create_key_concepts_tab(self):
        tab = self.create_tab("Key Concepts")
        content = """Key Concepts in Topic Modeling

1. Documents: Individual text entries in your corpus (e.g., articles, tweets, emails).

2. Words: The basic units that make up documents. Topic modeling often focuses on content words and filters out common words (stop words).

3. Corpus: The entire collection of documents being analyzed.

4. Topics: Hidden themes represented by collections of words. Each topic is essentially a probability distribution over words.

5. Topic Distribution: The proportion of each topic in a document. Every document is considered to be a mixture of topics.

6. Word Distribution: The probability of each word appearing in a given topic.

7. Term Frequency (TF): How often a word appears in a document.

8. Inverse Document Frequency (IDF): A measure of how important a word is to a document in a corpus.

9. TF-IDF: A numerical statistic that reflects how important a word is to a document in a corpus.

10. Dimensionality Reduction: The process of reducing the number of random variables under consideration, often used in topic modeling to identify the most important themes."""
        self.create_scrolled_text(tab, content)
    
    def create_lda_tab(self):
        tab = self.create_tab("LDA")
        content = """Latent Dirichlet Allocation (LDA)

LDA is one of the most popular algorithms for topic modeling, developed by Blei, Ng, and Jordan in 2003.

Key features of LDA:
1. Generative probabilistic model
2. Assumes documents are produced by a statistical process
3. Unsupervised learning algorithm

Key assumptions:
1. Each document is a mixture of topics
2. Each topic is a mixture of words

LDA Process:
1. For each document in the corpus:
   a. Choose a topic distribution
   b. For each word in the document:
      - Choose a topic from the document's topic distribution
      - Choose a word from that topic's word distribution

2. The "latent" part refers to the hidden topic structure uncovered by the algorithm
3. "Dirichlet" refers to the Dirichlet distribution used as a prior for topic distributions

Advantages of LDA:
1. Unsupervised, requiring no labeled training data
2. Can handle large document collections efficiently
3. Produces interpretable topics
4. Allows documents to belong to multiple topics, reflecting natural language complexity

Challenges:
1. Choosing the optimal number of topics
2. Dealing with polysemy and synonymy
3. Handling short texts with limited context
4. Interpreting and labeling discovered topics
5. Evaluating model quality (can be subjective)

Despite these challenges, LDA remains a valuable tool in text analysis, providing insights into large document collections that would be difficult to obtain through manual analysis."""
        self.create_scrolled_text(tab, content)
    
    def create_demo_tab(self):
        tab = self.create_tab("Interactive Demo")
        
        frame = ttk.Frame(tab)
        frame.pack(padx=10, pady=10)
        
        ttk.Label(frame, text="Number of Topics:").grid(row=0, column=0, padx=5, pady=5)
        self.num_topics_var = tk.StringVar(value="3")
        num_topics_entry = ttk.Entry(frame, textvariable=self.num_topics_var, width=5)
        num_topics_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(frame, text="Generate Topics", command=self.generate_topics).grid(row=0, column=2, padx=5, pady=5)
        
        self.result_text = scrolledtext.ScrolledText(tab, wrap=tk.WORD, width=110, height=35)  # Increased size
        self.result_text.pack(padx=10, pady=10)
    
    def generate_topics(self):
        try:
            num_topics = int(self.num_topics_var.get())
            if num_topics < 2 or num_topics > 5:
                raise ValueError("Number of topics must be between 2 and 5")
            
            topics = create_topic_model(corpus, num_topics)
            
            result = "Sample Corpus (Short News Articles):\n\n"
            for idx, doc in enumerate(corpus, 1):
                result += f"Article {idx}:\n{doc}\n\n"  # Display full article
            
            result += "Generated Topics:\n"
            for idx, topic in enumerate(topics, 1):
                result += f"Topic {idx}: {', '.join(topic)}\n"
            
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result)
            self.result_text.config(state=tk.DISABLED)
        
        except ValueError as e:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {str(e)}")
            self.result_text.config(state=tk.DISABLED)
    
    def create_quiz_tab(self):
        tab = self.create_tab("Quiz")
        
        frame = ttk.Frame(tab)
        frame.pack(padx=10, pady=10)
        
        ttk.Label(frame, text="Q1: What is the main purpose of Topic Modeling?").grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        self.q1_var = tk.StringVar()
        ttk.Radiobutton(frame, text="Data visualization", variable=self.q1_var, value="a").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Radiobutton(frame, text="Discover hidden themes in text", variable=self.q1_var, value="b").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Radiobutton(frame, text="Sentiment analysis", variable=self.q1_var, value="c").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        ttk.Radiobutton(frame, text="Machine translation", variable=self.q1_var, value="d").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        
        ttk.Label(frame, text="Q2: Which are key concepts in Topic Modeling? (Select all that apply)").grid(row=5, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        self.q2_vars = [tk.BooleanVar() for _ in range(5)]
        ttk.Checkbutton(frame, text="Documents", variable=self.q2_vars[0]).grid(row=6, column=0, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(frame, text="Images", variable=self.q2_vars[1]).grid(row=7, column=0, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(frame, text="Topics", variable=self.q2_vars[2]).grid(row=8, column=0, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(frame, text="Topic Distribution", variable=self.q2_vars[3]).grid(row=9, column=0, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(frame, text="Sentences", variable=self.q2_vars[4]).grid(row=10, column=0, sticky="w", padx=5, pady=2)
        
        ttk.Button(frame, text="Submit Quiz", command=self.submit_quiz).grid(row=11, column=0, columnspan=2, pady=10)
        
        self.quiz_result = tk.StringVar()
        ttk.Label(frame, textvariable=self.quiz_result).grid(row=12, column=0, columnspan=2, pady=5)
    
    def submit_quiz(self):
        score = 0
        if self.q1_var.get() == "b":
            score += 1
        
        correct_q2 = [True, False, True, True, False]
        if all(var.get() == correct for var, correct in zip(self.q2_vars, correct_q2)):
            score += 1
        
        self.quiz_result.set(f"Your score: {score}/2")

if __name__ == "__main__":
    root = tk.Tk()
    app = TopicModelingApp(root)
    root.mainloop()