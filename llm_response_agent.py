import os
import requests
from dotenv import load_dotenv
from mcp.protocol import MCPMessage
from pathlib import Path

# Load environment variables
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistralai/mistral-7b-instruct"  # ✅ You can change this

print("🔐 Loaded OPENROUTER_API_KEY:", "✅" if API_KEY else "❌ MISSING")


class LLMResponseAgent:
    def __init__(self, name="LLMResponseAgent"):
        self.name = name

    def generate_answer(self, retrieval_msg):
        try:
            query = retrieval_msg.payload["query"]
            sources = retrieval_msg.payload["retrieved_context"]
        except Exception as e:
            print(f"❌ Error parsing retrieval_msg payload: {e}")
            return MCPMessage(
                sender=self.name,
                receiver="CoordinatorAgent",
                msg_type="LLM_RESPONSE",
                trace_id=retrieval_msg.trace_id,
                payload={
                    "answer": "Invalid input format to LLM agent.",
                    "sources": []
                }
            )

        print("📄 Retrieved Sources:", sources)

        messages = [
            {"role": "system", "content": "You're a helpful assistant. Answer questions only using the provided context."},
            {"role": "user", "content": f"Context:\n{self._format_sources(sources)}\n\nQuestion:\n{query}"}
        ]

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.3
        }

        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            final_answer = result["choices"][0]["message"]["content"]

            return MCPMessage(
                sender=self.name,
                receiver="CoordinatorAgent",
                msg_type="LLM_RESPONSE",
                trace_id=retrieval_msg.trace_id,
                payload={
                    "answer": final_answer.strip(),
                    "sources": sources
                }
            )

        except requests.exceptions.RequestException as e:
            print("❌ Error contacting OpenRouter:", e)
            return MCPMessage(
                sender=self.name,
                receiver="CoordinatorAgent",
                msg_type="LLM_RESPONSE",
                trace_id=retrieval_msg.trace_id,
                payload={
                    "answer": "There was an error contacting the LLM.",
                    "sources": sources
                }
            )

    def _format_sources(self, chunks):
        if not chunks:
            return "No relevant context available."
        return "\n\n".join([f"- {chunk}" for chunk in chunks])
