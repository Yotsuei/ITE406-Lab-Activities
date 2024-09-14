import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter.messagebox as messagebox  # Add this import

# Download NLTK resources
nltk.download('punkt')
nltk.download('vader_lexicon')

# Initialize the books and reviews DataFrames
books_df = pd.DataFrame(columns=["Title", "Author", "Genre", "Availability"])
reviews_dict = {}  # Dictionary to hold book titles and associated reviews

# Function to add a new book
def add_book(title, author, genre):
    global books_df
    new_book = pd.DataFrame({"Title": [title], "Author": [author], "Genre": [genre], "Availability": ["Available"]})
    books_df = pd.concat([books_df, new_book], ignore_index=True)
    reviews_dict[title] = []  # Initialize an empty review list for the book
    update_treeview_books()
    
    # Clear input fields
    book_title_entry.delete(0, 'end')
    book_author_entry.delete(0, 'end')
    book_genre_entry.delete(0, 'end')
    
    # Show success message
    messagebox.showinfo("Success", f"Book '{title}' by {author} added successfully!")

# Function to search for books
def search_books(query):
    global books_df
    results = books_df[
        books_df['Title'].str.contains(query, case=False) |
        books_df['Author'].str.contains(query, case=False) |
        books_df['Genre'].str.contains(query, case=False)
    ]
    
    if not results.empty:
        result_str = results.to_string(index=False)
        messagebox.showinfo("Search Results", f"Books matching '{query}':\n{result_str}")
    else:
        messagebox.showinfo("No Results", f"No books found for query: '{query}'")
    
    # Clear search input
    search_entry.delete(0, 'end')

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to analyze review sentiment for a book
def analyze_review_sentiment(book_title, review):
    if book_title in reviews_dict:
        reviews_dict[book_title].append(review)  # Add the review to the book's review list
        sentiment_scores = sid.polarity_scores(review)
        result = f"Review Sentiment Analysis for '{book_title}': {sentiment_scores}\n"
        if sentiment_scores['compound'] >= 0.05:
            result += "Overall Sentiment: Positive"
        elif sentiment_scores['compound'] <= -0.05:
            result += "Overall Sentiment: Negative"
        else:
            result += "Overall Sentiment: Neutral"
        
        # Clear input fields
        review_book_title_entry.delete(0, 'end')
        review_entry.delete(0, 'end')
        
        # Show result in a pop-up
        messagebox.showinfo("Sentiment Analysis Result", result)
    else:
        messagebox.showerror("Error", f"Book '{book_title}' does not exist.")

# Function to add a borrower
borrowers_df = pd.DataFrame(columns=["Borrower Name", "Book Title", "Borrow Date", "Due Date"])

def add_borrower(borrower_name, book_title):
    global borrowers_df, books_df
    book_exists = books_df[books_df['Title'] == book_title]

    if not book_exists.empty:  # If the book exists
        if books_df.loc[books_df['Title'] == book_title, 'Availability'].values[0] == "Available":
            borrow_date = datetime.date.today()
            due_date = borrow_date + datetime.timedelta(days=14)  # 2 weeks borrowing period
            new_borrower = pd.DataFrame({
                "Borrower Name": [borrower_name],
                "Book Title": [book_title],
                "Borrow Date": [borrow_date],
                "Due Date": [due_date]
            })
            borrowers_df = pd.concat([borrowers_df, new_borrower], ignore_index=True)
            books_df.loc[books_df['Title'] == book_title, 'Availability'] = "Borrowed"
            update_treeview_borrowers()
            update_treeview_books()
            
            # Clear input fields
            borrower_name_entry.delete(0, 'end')
            borrow_book_entry.delete(0, 'end')
            
            # Show success message
            messagebox.showinfo("Success", f"'{book_title}' has been borrowed by {borrower_name}.")
        else:
            messagebox.showwarning("Not Available", f"Sorry, '{book_title}' is currently not available.")
    else:
        messagebox.showerror("Error", f"Book '{book_title}' does not exist in the library.")

# Function to return a book
def return_book(borrower_name, book_title):
    global borrowers_df, books_df
    borrowers_df = borrowers_df[
        ~(borrowers_df['Borrower Name'] == borrower_name) & (borrowers_df['Book Title'] == book_title)
    ]
    books_df.loc[books_df['Title'] == book_title, 'Availability'] = "Available"
    update_treeview_borrowers()
    update_treeview_books()
    
    # Clear input fields
    return_borrower_name_entry.delete(0, 'end')
    return_book_title_entry.delete(0, 'end')
    
    # Show success message
    messagebox.showinfo("Success", f"'{book_title}' has been returned by {borrower_name} and is now available.")

# ====================================================== Event Handlers =======================================================

def on_add_book():
    add_book(book_title_entry.get(), book_author_entry.get(), book_genre_entry.get())
def on_search():
    search_books(search_entry.get())
def on_analyze_sentiment():
    analyze_review_sentiment(review_book_title_entry.get(), review_entry.get())
def on_borrow_book():
    add_borrower(borrower_name_entry.get(), borrow_book_entry.get())
def on_return_book():
    return_book(return_borrower_name_entry.get(), return_book_title_entry.get())

# Function to update books Treeview
def update_treeview_books():
    for i in treeview_books.get_children():
        treeview_books.delete(i)
    for idx, row in books_df.iterrows():
        treeview_books.insert("", "end", values=list(row))

# Function to update borrowers Treeview
def update_treeview_borrowers():
    for i in treeview_borrowers.get_children():
        treeview_borrowers.delete(i)
    for idx, row in borrowers_df.iterrows():
        treeview_borrowers.insert("", "end", values=list(row))

# ========================================================== UI Code ==========================================================

app = ttk.Window(themename="superhero")
app.title("Library Management System")

windowWidth = 900
windowHeight = 600
displayWidth = app.winfo_screenwidth()
displayHeight = app.winfo_screenheight()

left = displayWidth / 2 - windowWidth / 2
top = displayHeight / 2 - windowHeight / 2
app.geometry(f"{windowWidth}x{windowHeight}+{int(left)}+{int(top)}")

# Output label
output_text = ttk.StringVar()
output_label = ttk.Label(app, textvariable=output_text, bootstyle="info", wraplength=500)
output_label.pack(pady=10)

# Notebook for Tabs
notebook = ttk.Notebook(app)
notebook.pack(fill="both", expand=True)

# Scrollable Tab 1: Book and Borrowing Management
frame_tab1 = ttk.Frame(notebook)
notebook.add(frame_tab1, text="Manage Books & Borrowing")

# Create a canvas and a scrollbar for Tab 1
canvas = ttk.Canvas(frame_tab1)
scrollbar = ttk.Scrollbar(frame_tab1, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

# Configure canvas and scrollbar
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Pack the canvas and scrollbar in the frame
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Enable scroll wheel for scrolling
def on_mouse_wheel(event):
    canvas.yview_scroll(-1 * int(event.delta / 120), "units")

# Bind mouse wheel scroll to canvas
scrollable_frame.bind_all("<MouseWheel>", on_mouse_wheel)

# Use grid layout for the scrollable frame to better use space horizontally
scrollable_frame.columnconfigure(1, weight=1)
scrollable_frame.columnconfigure(2, weight=1)

# Add a Book Section
ttk.Label(scrollable_frame, text="Add a Book", bootstyle="success").grid(row=0, column=0, pady=10, padx=10, sticky="w")
ttk.Label(scrollable_frame, text="Title:").grid(row=1, column=0, padx=10, sticky="w")
book_title_entry = ttk.Entry(scrollable_frame)
book_title_entry.grid(row=1, column=1, padx=10, sticky="ew")

ttk.Label(scrollable_frame, text="Author:").grid(row=2, column=0, padx=10, sticky="w")
book_author_entry = ttk.Entry(scrollable_frame)
book_author_entry.grid(row=2, column=1, padx=10, sticky="ew")

ttk.Label(scrollable_frame, text="Genre:").grid(row=3, column=0, padx=10, sticky="w")
book_genre_entry = ttk.Entry(scrollable_frame)
book_genre_entry.grid(row=3, column=1, padx=10, sticky="ew")

ttk.Button(scrollable_frame, text="Add Book", command=on_add_book).grid(row=4, column=1, pady=10, padx=10, sticky="e")

# Borrow Book Section
ttk.Label(scrollable_frame, text="Borrow Book", bootstyle="success").grid(row=5, column=0, pady=10, padx=10, sticky="w")
ttk.Label(scrollable_frame, text="Borrower Name:").grid(row=6, column=0, padx=10, sticky="w")
borrower_name_entry = ttk.Entry(scrollable_frame)
borrower_name_entry.grid(row=6, column=1, padx=10, sticky="ew")

ttk.Label(scrollable_frame, text="Book Title:").grid(row=7, column=0, padx=10, sticky="w")
borrow_book_entry = ttk.Entry(scrollable_frame)
borrow_book_entry.grid(row=7, column=1, padx=10, sticky="ew")

ttk.Button(scrollable_frame, text="Borrow", command=on_borrow_book).grid(row=8, column=1, pady=10, padx=10, sticky="e")

# Return Book Section
ttk.Label(scrollable_frame, text="Return Book", bootstyle="success").grid(row=9, column=0, pady=10, padx=10, sticky="w")
ttk.Label(scrollable_frame, text="Borrower Name:").grid(row=10, column=0, padx=10, sticky="w")
return_borrower_name_entry = ttk.Entry(scrollable_frame)
return_borrower_name_entry.grid(row=10, column=1, padx=10, sticky="ew")

ttk.Label(scrollable_frame, text="Book Title:").grid(row=11, column=0, padx=10, sticky="w")
return_book_title_entry = ttk.Entry(scrollable_frame)
return_book_title_entry.grid(row=11, column=1, padx=10, sticky="ew")

ttk.Button(scrollable_frame, text="Return Book", command=on_return_book).grid(row=12, column=1, pady=10, padx=10, sticky="e")

# Search for Books Section
ttk.Label(scrollable_frame, text="Search Books", bootstyle="success").grid(row=13, column=0, pady=10, padx=10, sticky="w")
ttk.Label(scrollable_frame, text="Enter search query:").grid(row=14, column=0, padx=10, sticky="w")
search_entry = ttk.Entry(scrollable_frame)
search_entry.grid(row=14, column=1, padx=10, sticky="ew")

ttk.Button(scrollable_frame, text="Search", command=on_search).grid(row=15, column=1, pady=10, padx=10, sticky="e")

# Review Sentiment Analysis Section
ttk.Label(scrollable_frame, text="Enter a Review for Sentiment Analysis", bootstyle="success").grid(row=16, column=0, pady=10, padx=10, sticky="w")
ttk.Label(scrollable_frame, text="Book Title:").grid(row=17, column=0, padx=10, sticky="w")
review_book_title_entry = ttk.Entry(scrollable_frame)
review_book_title_entry.grid(row=17, column=1, padx=10, sticky="ew")

ttk.Label(scrollable_frame, text="Review:").grid(row=18, column=0, padx=10, sticky="w")
review_entry = ttk.Entry(scrollable_frame)
review_entry.grid(row=18, column=1, padx=10, sticky="ew")

ttk.Button(scrollable_frame, text="Analyze Review Sentiment", command=on_analyze_sentiment).grid(row=19, column=1, pady=10, padx=10, sticky="e")

# Tab 2: Book and Borrower Lists
frame_lists = ttk.Frame(notebook)
notebook.add(frame_lists, text="Book & Borrower Lists")

frame_lists.columnconfigure(0, weight=1)

# Treeview for Books
ttk.Label(frame_lists, text="Books List", bootstyle="info").grid(row=0, column=0, pady=5, sticky="w")
treeview_books = ttk.Treeview(frame_lists, columns=("Title", "Author", "Genre", "Availability"), show='headings')
treeview_books.heading("Title", text="Title")
treeview_books.heading("Author", text="Author")
treeview_books.heading("Genre", text="Genre")
treeview_books.heading("Availability", text="Availability")
treeview_books.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

# Treeview for Borrowers
ttk.Label(frame_lists, text="Borrowers List", bootstyle="info").grid(row=2, column=0, pady=5, sticky="w")
treeview_borrowers = ttk.Treeview(frame_lists, columns=("Borrower Name", "Book Title", "Borrow Date", "Due Date"), show='headings')
treeview_borrowers.heading("Borrower Name", text="Borrower Name")
treeview_borrowers.heading("Book Title", text="Book Title")
treeview_borrowers.heading("Borrow Date", text="Borrow Date")
treeview_borrowers.heading("Due Date", text="Due Date")
treeview_borrowers.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

# Adjust column weights for equal expansion 
scrollable_frame.columnconfigure(1, weight=1)
frame_lists.columnconfigure(0, weight=1)

# Run the app
app.mainloop()