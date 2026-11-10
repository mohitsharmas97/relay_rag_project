import os
import re
from typing import List, Dict, Any
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Map filenames (both custom and arXiv IDs) to clean titles for citations
PAPER_NAME_MAP = {
    "attention": "Attention Is All You Need",
    "1706.03762v7": "Attention Is All You Need",
    "moshi": "Moshi: a speech-text foundation model for real-time dialogue",
    "2410.00037v1": "Moshi: a speech-text foundation model for real-time dialogue",
    "dynamic_chunking": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
    "2507.07955v2": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"
}

class PaperRetriever:
    """Handles paper chunking, embedding generation, indexing, and retrieval."""
    
    def __init__(self, papers_dir: str = "data/papers"):
        self.papers_dir = papers_dir
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents: List[Dict[str, Any]] = []
        self.index = None

    def load_and_chunk_papers(self, piece_size: int = 500, overlap: int = 50):
        """Extracts text from PDF papers in data/papers/ and chunks them."""
        self.documents = []
        if not os.path.exists(self.papers_dir):
            os.makedirs(self.papers_dir, exist_ok=True)
            
        pdf_files = [f for f in os.listdir(self.papers_dir) if f.endswith(".pdf")]
        
        for file_name in pdf_files:
            raw_key = file_name.replace(".pdf", "").lower()
            paper_title = PAPER_NAME_MAP.get(raw_key, raw_key.capitalize())
            
            reader = PdfReader(os.path.join(self.papers_dir, file_name))
            full_text = ""
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                full_text += f"\n[Page {page_idx + 1}] " + text

            words = full_text.split()
            for i in range(0, len(words), piece_size - overlap):
                chunk_text = " ".join(words[i:i + piece_size])
                if len(chunk_text.strip()) > 50:
                    self.documents.append({
                        "paper": paper_title,
                        "text": chunk_text,
                        "chunk_id": len(self.documents)
                    })

        if self.documents:
            embeddings = self.model.encode([doc["text"] for doc in self.documents], show_progress_bar=False)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings)
            self.index.add(np.array(embeddings).astype("float32"))

    def retrieve(self, query: str, pieces_fetched: int = 3, rerank: bool = False) -> List[Dict[str, Any]]:
        """Retrieves top relevant passages from the vector store."""
        if not self.index or len(self.documents) == 0:
            return []
            
        fetch_k = pieces_fetched * 2 if rerank else pieces_fetched
        query_vector = self.model.encode([query])
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(np.array(query_vector).astype("float32"), fetch_k)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["score"] = float(score)
                results.append(doc)

        if rerank and results:
            query_words = set(re.findall(r'\w+', query.lower()))
            for r in results:
                doc_words = set(re.findall(r'\w+', r["text"].lower()))
                overlap = len(query_words.intersection(doc_words))
                r["rerank_score"] = r["score"] + (overlap * 0.05)
            results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:pieces_fetched]
            
        return results[:pieces_fetched]