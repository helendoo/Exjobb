from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import pandas as pd
import os
import re

# === Set API Key and Model ===
os.environ["GROQ_API_KEY"] = "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"
llm = ChatGroq(model="llama-3.3-70b-versatile", verbose=True)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Load error knowledge base ===
system_prompt = (
    "Only answer using the provided context. If uncertain, say 'I don't know.' "
    "Answer questions strictly related to machine faults, error codes, or anomaly explanations."
)

storage_context = StorageContext.from_defaults(persist_dir="combined_index")
combined_index = load_index_from_storage(storage_context)
chat_engine = combined_index.as_chat_engine(chat_mode="context", llm=llm, system_prompt=system_prompt)






# === Load anomaly list for display ===
try:
    anomaly_df = pd.read_csv("MWD_Anomaly_Results.csv")
    anomalies = anomaly_df[anomaly_df['anomaly_label'] == 1][['Hole ID', 'Start Hole Time']].head()
except Exception:
    anomalies = None

# === LangGraph Chat State ===
class ChatState(TypedDict):
    user_input: Optional[str]
    response: Optional[str]
    end: Optional[bool]

# === Node: User Input ===
def ask_node(state: ChatState) -> ChatState:
    user_input = input("You: ")
    if user_input.lower() == "q":
        return {"user_input": None, "end": True}
    return {"user_input": user_input, "end": False}

# === Node: Handle Chat and Special Commands ===
def chat_node(state: ChatState) -> ChatState:
    if state["user_input"] is None:
        return state

    user_input = state["user_input"].strip().lower()

    if user_input == "/show anomalies":
        if anomalies is not None and not anomalies.empty:
            print("\n\U0001F4CD Recent anomalies:")
            for _, row in anomalies.iterrows():
                print(f"- Hole ID {int(row['Hole ID'])} at {row['Start Hole Time']}")
        else:
            print("No anomaly data available.")
        return {"response": "", "end": False}


    # Fall back to original error knowledge base
    response = chat_engine.chat(state["user_input"])
    return {"response": response.response}

# === Node: Show Response ===
def respond_node(state: ChatState) -> ChatState:
    if state.get("response"):
        print("Bot:", state["response"])
    return {}

# === LangGraph Definition ===
builder = StateGraph(ChatState)
builder.add_node("ask", ask_node)
builder.add_node("chat", chat_node)
builder.add_node("respond", respond_node)
builder.set_entry_point("ask")
builder.add_edge("ask", "chat")
builder.add_edge("chat", "respond")
builder.add_conditional_edges("respond", lambda s: "END" if s.get("end") else "ask", {"ask": "ask", "END": END})

# === Run the Graph ===
graph = builder.compile()
graph.invoke({})
