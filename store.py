# vector_store/store.py

import faiss
import numpy as np

class FAISSVectorStore:
    def __init__(self, embedding_dim=384):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.text_chunks = []

    def add(self, embeddings, chunks):
        embeddings = np.array(embeddings).astype("float32")

        # 🛡️ Ensure it's always 2D (even if 1 embedding)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # 🧹 Sanity check
        if embeddings.shape[1] != self.index.d:
            print("❌ Embedding dimension mismatch. Skipping FAISS add.")
            return

        if embeddings.shape[0] != len(chunks):
            print("❌ Number of embeddings ≠ number of chunks. Skipping FAISS add.")
            return

        # ✅ Add to FAISS
        self.index.add(embeddings)
        self.text_chunks.extend(chunks)
        print(f"✅ Added {len(chunks)} chunks to FAISS")
    
    def search(self, query_embedding, top_k=5):
       if self.index.ntotal == 0:
           print("❌ FAISS index is empty. Nothing to search.")
           return []

       query_embedding = np.array([query_embedding]).astype("float32")
       D, I = self.index.search(query_embedding, top_k)

    # 🛡️ Filter out invalid indices
       results = []
       for i in I[0]:
          if 0 <= i < len(self.text_chunks):
             results.append(self.text_chunks[i])
       return results
