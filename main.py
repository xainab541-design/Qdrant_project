import json
from loguru import logger
from qdrant_manager import QdrantManager

def main():
    logger.info("Initializing Vector Search Pipeline...")
    
    # 1. Initialize Manager (this connects to Qdrant and creates the collection if needed)
    # Note: Requires Sentence-transformers ~450MB model download on first run
    manager = QdrantManager(
        collection_name="knowledge_base",
        vector_size=768 # sentence-transformers/all-mpnet-base-v2 is 768-dimensional
    )
    
    # 2. Load dummy data
    try:
        with open("dummy_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} records from dummy_data.json")
    except FileNotFoundError:
        logger.error("dummy_data.json not found. Please run generate_dummy_data.py first.")
        return

    # 3. Process, chunk, embed, and upsert
    manager.process_and_upsert(data, batch_size=32)
    
    # 4. Perform a test query
    query_text = "Looking for durable sports equipment for outdoor camping"
    # Given the dummy data, we might need a lower threshold to see a hit, or use a better query
    results = manager.query(query_text=query_text, top_k=3, score_threshold=0.2)
    
    logger.info("--- Search Results ---")
    for idx, res in enumerate(results, 1):
        logger.info(f"Result {idx} (Score: {res.score:.4f}):")
        logger.info(f"  Text: {res.text}")
        logger.info(f"  Category: {res.metadata.category}")
        logger.info(f"  Source URL: {res.metadata.source_url}")
        logger.info("-" * 40)
        
    logger.info("\nPipeline execution complete.")
    logger.info("You can view your data in the Qdrant Dashboard at: http://localhost:6333/dashboard")

if __name__ == "__main__":
    main()
