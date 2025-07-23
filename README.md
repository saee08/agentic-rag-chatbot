

A modular, agent-based **Retrieval-Augmented Generation (RAG)** chatbot that can answer user questions using uploaded documents in multiple formats. Built with **Streamlit**, **FAISS**, **OpenAI GPT**, and the **Model Context Protocol (MCP)**.

---

## 🚀 Features

-  Upload and parse documents (PDF, DOCX, PPTX, CSV, TXT, Markdown)
-  Multi-agent architecture (Ingestion → Retrieval → LLM Response)
-  Semantic search using embeddings and FAISS
-  LLM-powered answers with source context
-  Structured communication using MCP (Model Context Protocol)
-  Built with modular design for easy scaling and testing

---

## 🧠 Architecture

### Agents:
- **IngestionAgent**: Parses documents and splits them into semantic chunks.
- **RetrievalAgent**: Embeds and indexes chunks using FAISS, retrieves relevant ones for a query.
- **LLMResponseAgent**: Uses top chunks + query to generate an answer via OpenAI GPT.
- **CoordinatorAgent**: Orchestrates all agents for a clean, single-point interface.

### MCP (Model Context Protocol):
All agents communicate via structured MCP message objects:
```json
{
  "sender": "RetrievalAgent",
  "receiver": "LLMResponseAgent",
  "type": "RETRIEVAL_RESULT",
  "trace_id": "abc-123",
  "payload": {
    "query": "What are the KPIs?",
    "retrieved_context": ["...", "..."]
  }
}


some of the input files:

📚 Test PDFs with Download Links + Questions
1. 📊 IBM Annual Report 2022
Download: IBM_Annual_Report_2022.pdf

What to Ask:

“What were IBM's total revenues in 2022?”

“Summarize the CEO’s message.”

“Which business segments performed best?”

“What are the future goals outlined in the report?”

2. 📄 UN SDG Report (Sustainable Development Goals)
Download: UN_SDG_Report_2023.pdf

What to Ask:

“What progress has been made on Goal 13 (Climate Action)?”

“List challenges faced in implementing SDGs.”

“Which countries showed the most progress?”

3. 🧪 NASA Artemis Mission Summary
Download: NASA_Artemis_Plan.pdf

What to Ask:

“What are the objectives of the Artemis program?”

“Which year is Artemis III planned?”

“What is the Gateway?”

4. 👩‍🎓 AI Research Paper (Transformer Models)
Download: Attention is All You Need

What to Ask:

“What is the purpose of the self-attention mechanism?”

“Explain the transformer model architecture.”

“How does this model differ from RNNs?”


SAMPLE QUESTIONS :
After uploading documents, try asking:

Resume: “What are my technical skills?”

Research Paper: “What is the proposed methodology?”

Project Report: “What KPIs are discussed?”

CSV: “What is the average of the column X?”


--------------------------------------------------------------------------
######tech stack used :#######
Streamlit – Frontend interface

SentenceTransformer – Embedding model

FAISS – Vector similarity search

OpenRouter API – GPT-like LLM access

Python – All agents and backend logic

MCP (Model Context Protocol) – Lightweight protocol for agent communication


-------------------------------------------------------
IngestionAgent: Parses files, splits into semantic chunks

RetrievalAgent: Converts chunks into embeddings, stores/retrieves using FAISS

LLMResponseAgent: Sends context + query to OpenRouter model (e.g., Mistral) and returns answer

CoordinatorAgent: Central orchestrator managing query → chunk → answer pipeline

# agentic-rag-chatbot

