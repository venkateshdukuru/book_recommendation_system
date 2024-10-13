import json
import os
from typing import Literal, Optional, List
from uuid import uuid4
from fastapi import FastAPI, HTTPException
import random
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from mangum import Mangum
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class Book(BaseModel):
    name: str
    genre: Literal["fiction", "non-fiction"]
    price: float
    book_id: Optional[str] = uuid4().hex

BOOKS_FILE = "books.json"
BOOKS = []

if os.path.exists(BOOKS_FILE):
    with open(BOOKS_FILE, "r") as f:
        BOOKS = json.load(f)

app = FastAPI()
handler = Mangum(app)

def encode_genre(genre: str) -> int:
    """Encode genre as a numerical feature."""
    return 1 if genre == "fiction" else 0

def prepare_data(books):
    """Prepare feature matrix for KNN using TF-IDF for names and encoding for genre."""
    names = [book["name"] for book in books]
    genres = [encode_genre(book["genre"]) for book in books]
    
    # Convert names to TF-IDF features
    vectorizer = TfidfVectorizer()
    name_features = vectorizer.fit_transform(names).toarray()
    
    # Combine name features with genre
    features = np.hstack((name_features, np.array(genres).reshape(-1, 1)))
    return features

@app.get("/")
async def root():
    return {"message": "Welcome to my bookstore app!"}

@app.get("/random-book")
async def random_book():
    return random.choice(BOOKS)

@app.get("/list-books")
async def list_books():
    return {"books": BOOKS}

@app.get("/book_by_index/{index}")
async def book_by_index(index: int):
    if index < len(BOOKS):
        return BOOKS[index]
    else:
        raise HTTPException(404, f"Book index {index} out of range ({len(BOOKS)}).")

@app.post("/add-book")
async def add_book(book: Book):
    book.book_id = uuid4().hex
    json_book = jsonable_encoder(book)
    BOOKS.append(json_book)

    with open(BOOKS_FILE, "w") as f:
        json.dump(BOOKS, f)

    return {"book_id": book.book_id}

@app.get("/get-book")
async def get_book(book_id: str):
    for book in BOOKS:
        if book["book_id"] == book_id:
            return book

    raise HTTPException(404, f"Book ID {book_id} not found in database.")

@app.get("/recommend-books/{book_name}")
async def recommend_books(book_name: str) -> List[Book]:
    # Prepare the feature matrix
    features = prepare_data(BOOKS)

    # Fit the KNN model
    knn = NearestNeighbors(n_neighbors=3)  # Adjust n_neighbors as needed
    knn.fit(features)

    # Normalize the book name for case-insensitive matching
    normalized_book_name = book_name.lower()

    # Find the index of the book that matches the provided name
    book_index = next((index for index, book in enumerate(BOOKS) if book["name"].lower() == normalized_book_name), None)

    if book_index is None:
        raise HTTPException(404, f"No book found with the name '{book_name}'.")

    # Get the features of the book to find recommendations for
    query_features = features[book_index].reshape(1, -1)

    # Get the indices of the nearest neighbors
    distances, indices = knn.kneighbors(query_features)

    # Collect recommended books, excluding the original book
    recommendations = [BOOKS[i] for i in indices[0] if i != book_index]

    if not recommendations:
        raise HTTPException(404, f"No recommendations found for book '{book_name}'.")

    return recommendations


""" @app.get("/recommend-books/{book_id}")
async def recommend_books(book_id: str) -> List[Book]:
    # Prepare the feature matrix
    features = prepare_data(BOOKS)

    # Fit the KNN model
    knn = NearestNeighbors(n_neighbors=3)  # Adjust n_neighbors as needed
    knn.fit(features)

    # Find the index of the requested book
    book_index = next((index for index, book in enumerate(BOOKS) if book["book_id"] == book_id), None)

    if book_index is None:
        raise HTTPException(404, f"Book ID {book_id} not found.")

    # Get the features of the book to find recommendations for
    query_features = features[book_index].reshape(1, -1)

    # Get the indices of the nearest neighbors
    distances, indices = knn.kneighbors(query_features)

    # Collect recommended books, excluding the original book
    recommendations = [BOOKS[i] for i in indices[0] if i != book_index]

    # If no recommendations based on KNN, recommend based on genre and name
    if not recommendations:
        genre = BOOKS[book_index]["genre"]
        name = BOOKS[book_index]["name"]
        recommendations = [
            book for book in BOOKS 
            if (book["genre"] == genre or name.lower() in book["name"].lower()) and book["book_id"] != book_id
        ]

    return recommendations"""

