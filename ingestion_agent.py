# agents/ingestion_agent.py
import uuid
from file_parser.parser import parse_file
from mcp.protocol import MCPMessage
from sentence_transformers import SentenceTransformer
import torch

# Simple chunking function
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join
        
        
        (words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

class IngestionAgent:
    def __init__(self):
        self.name = "IngestionAgent" 
        print(f"✅ Initialized {self.name}")

        # Automatically choose device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📦 Loading embedding model on: {device}")
        
        # SAFELY load without triggering meta tensor bug
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # don't pass device here
        self.model.to(device)

    def process_documents(self, file_paths):
        all_chunks = []
        for path in file_paths:
            try:
                raw_text = parse_file(path)
                print(f"✅ Extracted text from {path[:40]}...")

                chunks = chunk_text(raw_text)
                print(f"✅ {len(chunks)} chunks created.")
                
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error parsing {path}: {e}")
        
        trace_id = str(uuid.uuid4())
        message = MCPMessage(
            sender=self.name,
            receiver="RetrievalAgent",
            msg_type="INGESTION_COMPLETE",
            trace_id=trace_id,
            payload={"chunks": all_chunks}
        )
        return message
