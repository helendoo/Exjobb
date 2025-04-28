from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import pandas as pd
import os

#langGraph Framework + groq + Ollama


# LLM 
os.environ["GROQ_API_KEY"] = "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"  
llm = ChatGroq(model="llama-3.3-70b-versatile", verbose=True)

# Embedding 
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Load Data & Build Index ===
df = pd.read_csv("errors.csv") 
documents = [Document(text=row.to_string()) for _, row in df.iterrows()]
index = VectorStoreIndex.from_documents(documents)
chat_engine = index.as_chat_engine(chat_mode="context", llm=llm, system_prompt = ("Only answer using provided context. If uncertain, say you don't know."))

# State Schema 
class ChatState(TypedDict):
    user_input: Optional[str]
    response: Optional[str]
    end: Optional[bool]

# Graph Nodes 
def ask_node(state: ChatState) -> ChatState:
    user_input = input("You: ")
    if user_input.lower() == "q":
        return {"user_input": None, "end": True}
    return {"user_input": user_input, "end": False}


def chat_node(state: ChatState) -> ChatState:
    if state["user_input"] is None:
        return state
    response = chat_engine.chat(state["user_input"])
    return {"response": response.response}


def respond_node(state: ChatState) -> ChatState:
    if state.get("response"):
        print("Bot:", state["response"])
    return {}

# Build Graph
builder = StateGraph(ChatState)
builder.add_node("ask", ask_node)
builder.add_node("chat", chat_node)
builder.add_node("respond", respond_node)

builder.set_entry_point("ask")
builder.add_edge("ask", "chat")
builder.add_edge("chat", "respond")
builder.add_conditional_edges("respond", lambda s: "END" if s.get("end") else "ask", {"ask": "ask", "END": END})


graph = builder.compile()
graph.invoke({})

""" img_bytes = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(img_bytes) """