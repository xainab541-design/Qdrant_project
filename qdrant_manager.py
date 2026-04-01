from typing import List, Optional, Dict, Any
import uuid
from loguru import logger
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from sentence_transformers import SentenceTransformer

# --- Pydantic Models for Data Validation ---

class DocumentMetadata(BaseModel):
    category: str
    source_url: str
    timestamp: str
    price: Optional[float] = None
    original_id: str
    chunk_index: int

class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: DocumentMetadata

class SearchResult(BaseModel):
    text: str
    score: float
    metadata: DocumentMetadata

# --- QdrantManager Class ---

class QdrantManager:
    """
    Manages Qdrant collections, data chunking, embeddings generation, 
    and vector search operations.
    """
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "knowledge_base",
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        vector_size: int = 768,
        chunk_size: int = 200,
        chunk_overlap: int = 50
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        logger.info(f"Connecting to Qdrant at {host}:{port}")
        self.client = QdrantClient(host=host, port=port)
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Creates the collection if it does not already exist."""
        collections_response = self.client.get_collections()
        collection_names = [col.name for col in collections_response.collections]
        
        if self.collection_name not in collection_names:
            logger.info(f"Creating collection '{self.collection_name}' with vector size {self.vector_size}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest_models.VectorParams(
                    size=self.vector_size,
                    distance=rest_models.Distance.COSINE
                )
            )
        else:
            logger.info(f"Collection '{self.collection_name}' already exists.")

    def chunk_text(self, text: str) -> List[str]:
        """
        Simple recursive character splitter.
        Splits text into chunks of roughly `chunk_size` characters with `chunk_overlap`.
        """
        words = text.split(" ")
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep overlap words
                overlap_words = []
                overlap_length = 0
                for w in reversed(current_chunk):
                    if overlap_length + len(w) + 1 > self.chunk_overlap:
                        break
                    overlap_words.insert(0, w)
                    overlap_length += len(w) + 1
                
                current_chunk = overlap_words
                current_length = overlap_length
            
            current_chunk.append(word)
            current_length += len(word) + 1
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def process_and_upsert(self, data: List[Dict[str, Any]], batch_size: int = 32):
        """
        Chunks the input data, generates embeddings, and cleanly upserts to Qdrant in batches.
        """
        logger.info(f"Processing {len(data)} documents...")
        all_chunks: List[DocumentChunk] = []
        
        for item in data:
            full_text = f"{item.get('title', '')}. {item.get('content', '')}"
            text_chunks = self.chunk_text(full_text)
            
            for idx, chunk_text in enumerate(text_chunks):
                metadata = DocumentMetadata(
                    category=item.get("category", "Unknown"),
                    source_url=item.get("source_url", ""),
                    timestamp=item.get("timestamp", ""),
                    price=item.get("price"),
                    original_id=item.get("id", str(uuid.uuid4())),
                    chunk_index=idx
                )
                
                doc_chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=chunk_text,
                    metadata=metadata
                )
                all_chunks.append(doc_chunk)

        logger.info(f"Generated {len(all_chunks)} chunks from {len(data)} documents.")
        
        # Upsert in batches
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            texts = [chunk.text for chunk in batch]
            
            # Generate embeddings
            embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
            
            points = []
            for j, chunk in enumerate(batch):
                payload = chunk.metadata.model_dump()
                payload["text"] = chunk.text  # Store original text in payload for retrieval
                
                points.append(
                    rest_models.PointStruct(
                        id=chunk.chunk_id,
                        vector=embeddings[j],
                        payload=payload
                    )
                )
            
            logger.info(f"Upserting batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
        logger.info("Upsert complete.")

    def query(self, query_text: str, top_k: int = 5, score_threshold: float = 0.7) -> List[SearchResult]:
        """
        Converts a natural language query into an embedding and retrieves the Top-K results
        that meet the similarity score threshold.
        """
        logger.info(f"Searching for: '{query_text}'")
        query_vector = self.model.encode([query_text], convert_to_numpy=True)[0].tolist()
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold
        )
        
        results = []
        for hit in search_result:
            metadata_dict = hit.payload.copy()
            text = metadata_dict.pop("text", "")
            
            # Validate remaining payload is valid metadata
            metadata = DocumentMetadata(**metadata_dict)
            
            results.append(
                SearchResult(
                    text=text,
                    score=hit.score,
                    metadata=metadata
                )
            )
            
        logger.info(f"Found {len(results)} results with score > {score_threshold}")
        return results
