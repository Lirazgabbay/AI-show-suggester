# ShowSuggesterAI

## Overview
ShowSuggesterAI is an AI-powered TV show recommendation system that uses **embeddings and vector similarity** to generate highly accurate recommendations. 
This Python-based project employs **fuzzy matching**, **vector indexing**, and **image generation** to provide an enhanced user experience, even surpassing traditional recommendation engines.

## Features
- **User Input Processing:**
  - Accepts user-inputted TV shows and ensures correct spellings using **fuzzy matching** (`thefuzz`).
- **TV Show Embeddings:**
  - Uses OpenAI's **embedding API** to generate vector representations of TV show descriptions.
  - Saves embeddings locally using **pickle** to avoid redundant API calls.
- **Recommendation Engine:**
  - Computes **cosine similarity** to find the top **5 most relevant TV shows** based on user preferences.
  - Uses **vector search libraries** (`usearch` or `annoy`) for **efficient similarity lookups**.
- **Custom AI-Generated TV Shows:**
  - Generates **two unique TV show ideas** based on user preferences using **LightX image generation API**.
  - Displays the AI-generated TV show posters within the program.
- **Efficient Search with Vector Indexing:**
  - Uses **Redis vector search** (optional challenge) to improve retrieval speed for large datasets.
- **TDD (Test-Driven Development):**
  - Implemented unit tests with **pytest** to validate critical logic.
  - Ensured high **test coverage** using `pytest-cov`.


## Optimization & Scalability
- Uses **vector search libraries** like `usearch` for O(1) lookup instead of O(n) linear search.
- Can handle **large TV show datasets** efficiently.

