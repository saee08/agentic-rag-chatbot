# main.py
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent
import glob

if __name__ == "__main__":
    # 1. Simulate document ingestion
    files = glob.glob("data/uploads/*")
    ingestion = IngestionAgent()
    retrieval = RetrievalAgent()

    # 2. Process and pass chunks to RetrievalAgent
    msg = ingestion.process_documents(files)
    retrieval.receive_chunks(msg)

    # 3. User query
    user_query = "What are the KPIs mentioned?"
    response_msg = retrieval.handle_query(user_query, trace_id=msg.trace_id)

    print("\n🔍 Retrieved Chunks:")
    for i, chunk in enumerate(response_msg.payload["retrieved_context"], 1):
        print(f"{i}. {chunk[:100]}...")
    llm_agent = LLMResponseAgent()
    final_response = llm_agent.generate_answer(response_msg)

    print("\n💬 Final Answer:")
    print(final_response.payload["answer"])

    print("\n📚 Source Chunks:")
    for src in final_response.payload["sources"]:
        print("-", src[:100], "...")
