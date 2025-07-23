import uuid
import numpy as np
from mcp.protocol import MCPMessage
from vector_store.embedder import Embedder
from vector_store.store import FAISSVectorStore

class RetrievalAgent:
    def __init__(self, name="RetrievalAgent"):
        self.name = name
        self.embedder = Embedder()
        self.store = FAISSVectorStore()

    def receive_chunks(self, msg):
        chunks = msg.payload.get("chunks", [])

        if not chunks:
            print("❌ No chunks received. Aborting retrieval.")
            return

        # 🚨 Final strict sanitization: list of strings only
        clean_chunks = []
        for i, c in enumerate(chunks):
            if isinstance(c, str):
                clean_chunks.append(c)
            elif callable(c):
                print(f"❌ Skipping callable in chunk {i}: {c}")
            elif isinstance(c, (list, dict, tuple)):
                print(f"⚠️ Skipping complex data in chunk {i}: {type(c)}")
            else:
                print(f"🔁 Converting chunk {i} to string: {type(c)}")
                try:
                    clean_chunks.append(str(c))
                except Exception as e:
                    print(f"❌ Failed to convert chunk {i}: {e}")

        print(f"✅ Final cleaned chunk count: {len(clean_chunks)}")
        print("🔍 Example clean chunk:", clean_chunks[0][:100] if clean_chunks else "N/A")

        # Now safely embed
        embeddings = self.embedder.embed(clean_chunks)

        if embeddings is None or (isinstance(embeddings, np.ndarray) and embeddings.size == 0):
            print("❌ Embedding failed or returned empty.")
            return

        self.store.add(embeddings, clean_chunks)
        print("✅ Chunks embedded and stored successfully.")

    def handle_query(self, query, trace_id=None, top_k=5):
        if not query:
            print("❌ Empty query string.")
            return None

        query_embedding = self.embedder.embed([query])

        if query_embedding is None or len(query_embedding) == 0:
            print("❌ Query embedding failed.")
            return None

        top_chunks = self.store.search(query_embedding[0], top_k=top_k)

        return MCPMessage(
            sender=self.name,
            receiver="LLMResponseAgent",
            msg_type="RETRIEVAL_RESULT",
            trace_id=trace_id or str(uuid.uuid4()),
            payload={
                "retrieved_context": top_chunks,
                "query": query
            }
        )
