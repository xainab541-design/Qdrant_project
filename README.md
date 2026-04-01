# Qdrant Vector Search Pipeline

A high-performance semantic search pipeline that ingests data, generates high-dimensional mathematical embeddings using AI, and stores them in Qdrant for blazing-fast similarity searches.

## Overview
This project is a high-performance vector search pipeline designed to demonstrate the power of semantic search using Qdrant and Sentence-Transformers. Traditional keyword-based search engines often fail to capture the underlying meaning of user queries, returning results only when exact words match. In contrast, this pipeline implements an Artificial Intelligence-driven approach that understands context and semantics.

The system begins by generating a comprehensive mock e-commerce dataset containing products with rich metadata, such as categories, descriptions, URLs, and prices. Using the HuggingFace `sentence-transformers/all-mpnet-base-v2` model, the pipeline intelligently chunks the textual data and converts it into 768-dimensional mathematical vectors, known as embeddings. These embeddings encapsulate the deep semantic meaning of the text.

The generated vectors, along with their Pydantic-validated payload metadata, are cleanly batched and upserted into a localized, persistent Qdrant vector database running via Docker. Once digested, the system allows users to perform natural language queries. By converting a user's search query into a vector and calculating the Cosine Similarity against the stored database, the system instantly retrieves the top 'K' most contextually relevant products. Ultimately, this scalable architecture showcases how modern AI can build highly intuitive, meaning-aware search engines for real-world applications.

## Prerequisites
- Python 3.10+
- Docker Desktop

## Setup Instructions

1. **Install Dependencies**
   Install the necessary Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Qdrant Database**
   Run the following command to start Qdrant via Docker Compose (make sure Docker Desktop is running):
   ```bash
   docker compose up -d
   ```
   > The database will be accessible at [http://localhost:6333/dashboard](http://localhost:6333/dashboard).

## Usage

1. **Generate Dummy Data**
   Run the generator script to create a `.json` file containing 50 mock shop items:
   ```bash
   python generate_dummy_data.py
   ```

2. **Run the Pipeline**
   Execute the main file. This will initiate the `QdrantManager`, connect to the database, download the HuggingFace model, chunk the data, upsert the embeddings, and perform a test query:
   ```bash
   python main.py
   ```

## Project Structure
- `qdrant_manager.py`: Core logic for chunking texts, building embeddings, and connecting to Qdrant.
- `generate_dummy_data.py`: Creates `dummy_data.json` for testing.
- `main.py`: Entry point that coordinates ingestion and searching.
- `docker-compose.yml`: Local Docker environment configuration for a persistent Qdrant instance.
