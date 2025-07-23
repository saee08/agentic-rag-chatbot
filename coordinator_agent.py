# agents/coordinator_agent.py

from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent
import uuid

class CoordinatorAgent:
    def __init__(self, name="CoordinatorAgent"):
        self.name = name
        self.ingestion_agent = IngestionAgent()
        self.retrieval_agent = RetrievalAgent()
        self.llm_agent = LLMResponseAgent()

    # agents/coordinator_agent.py

def handle_user_query(self, file_paths, query):
    ingestion_msg = self.ingestion_agent.process_documents(file_paths)

    if not ingestion_msg.payload["chunks"]:
        return {
            "payload": {
                "answer": "❌ No content was extracted from the uploaded files.",
                "sources": []
            }
        }

    self.retrieval_agent.receive_chunks(ingestion_msg)

    if not query or not query.strip():
        return {
            "payload": {
                "answer": "⚠️ Please enter a valid question.",
                "sources": []
            }
        }

    trace_id = str(uuid.uuid4())  # ✅ generate here
    retrieval_msg = self.retrieval_agent.handle_query(query, trace_id)

    retrieval_msg = self.retrieval_agent.handle_query(query, trace_id)

    if retrieval_msg is None:
        print("❌ Retrieval failed. Aborting LLM call.")
    return {
        "payload": {
            "answer": "❌ Retrieval failed. Please check your query or input document.",
            "sources": []
        }
    }


    final_msg = self.llm_agent.generate_answer(retrieval_msg)
    return final_msg
