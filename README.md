# Book Recommendation System

A simple book recommendation API built with FastAPI. This system allows users to manage a collection of books and receive recommendations based on book names and genres using the K-Nearest Neighbors (KNN) algorithm.

## Features

- **Add Books**: Add new books to the collection with details such as name, genre, price, and a unique ID.
- **List Books**: Retrieve a list of all available books.
- **Get Book by Index**: Fetch book details using its index in the collection.
- **Get Book by ID**: Retrieve book details using its unique ID.
- **Random Book**: Get a random book from the collection.
- **Book Recommendations**: Recommend books based on a given book name, considering both the name and genre.

## Technologies Used

- **Python**: Programming language used for backend development.
- **FastAPI**: Framework for building APIs quickly and efficiently.
- **scikit-learn**: Machine learning library used for implementing the KNN algorithm.
- **Pydantic**: Data validation and settings management using Python type annotations.
- **Mangum**: Adapter for running FastAPI applications on AWS Lambda.

## Installation

1. **Clone the repository**:                                                                                                                                                                                         
   ```bash
   git clone https://github.com/yourusername/book-recommendation-system.git
   cd book-recommendation-system
2. **Create a virtual environment**:                                                                                                                                                                                 
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
3. **Install the required packages:**                                                                                                                                                                                
   pip install -r requirements.txt 
4. **Run the application:**                                                                                                                                                                                          
   uvicorn main:app --reload
   Visit http://127.0.0.1:8000 in your browser.
